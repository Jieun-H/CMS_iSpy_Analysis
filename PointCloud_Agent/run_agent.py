#!/usr/bin/env python3
"""
PointCloud_Agent: CMS AOD/AODSIM dataset -> iSpy ig -> point cloud (npz) + visualization (html).

Run on lxplus (el9 host shell, NOT inside cmssw-elN — the agent enters the right container itself):

  python3 run_agent.py /ZToMuMu_M-120To200_TuneCP5_14TeV-powheg-pythia8/Run3Summer21DRPremix-120X_mcRun3_2021_realistic_v6-v2/AODSIM

Convert ig files you already have (no CMSSW needed, also works locally):

  python3 run_agent.py --ig path/to/NAME_0.ig path/to/NAME_1.ig --name NAME --out ./output

See README.md for all options.
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agent import eras  # noqa: E402

CMSSW_DIR = os.path.join(HERE, "agent", "cmssw")


class AgentError(Exception):
    pass


def log(msg):
    print(f"[agent] {msg}", flush=True)


# ════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════

def user_default_dir(eos_sub, home_sub):
    """EOS user area if it exists (lxplus), otherwise $HOME."""
    user = os.environ.get("USER", "")
    eos = f"/eos/user/{user[:1]}/{user}"
    if user and os.path.isdir(eos):
        return os.path.join(eos, eos_sub)
    return os.path.join(os.path.expanduser("~"), home_sub)


def cmssw_default_dir():
    """AFS work area if it exists (more quota than $HOME), otherwise $HOME."""
    user = os.environ.get("USER", "")
    work = f"/afs/cern.ch/work/{user[:1]}/{user}"
    base = work if user and os.path.isdir(work) else os.path.expanduser("~")
    return os.path.join(base, "pca_cmssw")


def run(cmd, log_path=None, check=True):
    """Run a command, stream its output (and tee it to log_path). Returns (code, output)."""
    log("$ " + " ".join(cmd))
    out_lines = []
    logf = open(log_path, "a") if log_path else None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            out_lines.append(line)
            if logf:
                logf.write(line)
        code = proc.wait()
    finally:
        if logf:
            logf.close()
    if check and code != 0:
        raise AgentError(f"command failed (exit {code}): {' '.join(cmd)}"
                         + (f"\n  log: {log_path}" if log_path else ""))
    return code, "".join(out_lines)


def capture(cmd):
    """Run a short command and return stdout lines (raises on failure)."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if res.returncode != 0:
        raise AgentError(f"{' '.join(cmd)} failed:\n{res.stderr.strip()}")
    return [l.strip() for l in res.stdout.splitlines() if l.strip()]


def in_era(era, cmd):
    """Wrap a command so it runs in the era's OS container (cmssw-elN -- ...)."""
    if not era.container:
        return cmd
    wrapper = shutil.which(era.container) or f"/cvmfs/cms.cern.ch/common/{era.container}"
    if not os.path.exists(wrapper):
        raise AgentError(f"{era.container} not found — is this lxplus with /cvmfs/cms.cern.ch?")
    return [wrapper, "--"] + cmd


def git_info():
    try:
        commit = capture(["git", "-C", HERE, "rev-parse", "HEAD"])[0]
        dirty = bool(capture(["git", "-C", HERE, "status", "--porcelain", "--", "."]))
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


# ════════════════════════════════════════════════════════
#  plan
# ════════════════════════════════════════════════════════

def make_plan(args):
    plan = {"dataset": args.dataset}
    if args.ig:
        plan["name"] = args.name or os.path.splitext(os.path.basename(args.ig[0]))[0].rsplit("_", 1)[0]
        return plan

    info = eras.parse_dataset(args.dataset)
    plan["name"] = args.name or eras.default_name(args.dataset)
    plan["is_mc"] = info["is_mc"]
    if not info["tier"].startswith("AOD"):
        raise AgentError(f"Data tier is {info['tier']}; this agent expects AOD or AODSIM.")

    release = args.release
    if not release:
        try:
            releases = capture(["dasgoclient", "-query", f"release dataset={args.dataset}"])
            release = releases[0] if releases else None
        except (AgentError, FileNotFoundError) as e:
            log(f"DAS release lookup failed ({e}); falling back to the dataset name")
        release = release or eras.release_from_name(args.dataset)
    if not release:
        raise AgentError("Could not determine the CMSSW release; pass --release CMSSW_X_Y_Z")
    era = eras.era_for_release(release)

    gt = args.globaltag or (eras.mc_globaltag(args.dataset) if info["is_mc"] else era.data_globaltag)
    if not gt:
        raise AgentError("Could not determine the GlobalTag; pass --globaltag")

    plan.update({
        "das_release": release, "era": era.key, "cmssw": era.cmssw, "scram_arch": era.scram_arch,
        "container": era.container, "ispy_branch": era.ispy_branch,
        "services_branch": era.services_branch, "globaltag": gt, "era_validated": era.validated,
    })
    return plan


# ════════════════════════════════════════════════════════
#  step 1: AOD -> ig (lxplus)
# ════════════════════════════════════════════════════════

def step_ig(args, plan, out):
    if os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"):
        raise AgentError("Run the agent from the plain lxplus shell, not inside cmssw-elN.")
    era = eras.era_for_release(plan["das_release"])
    if not era.validated:
        log(f"NOTE: era '{era.key}' has not been validated end-to-end yet. {era.notes}")

    if subprocess.run(["voms-proxy-info", "--exists", "--valid", "0:30"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise AgentError("No valid grid proxy (or <30 min left). Run:\n"
                         "  voms-proxy-init --rfc --voms cms --valid 168:00")

    logs = os.path.join(out, "logs")
    base = os.path.abspath(args.cmssw_base)
    area = os.path.join(base, era.cmssw)

    # 1. CMSSW area + ISpy build (first run per era takes 10-20 min)
    log(f"CMSSW area: {area}  (ISpy {era.ispy_branch})")
    run(in_era(era, ["bash", os.path.join(CMSSW_DIR, "setup_cmssw.sh"), base, era.cmssw,
                     era.scram_arch, era.ispy_branch, era.services_branch]),
        log_path=os.path.join(logs, "setup_cmssw.log"))

    # 2. input files
    files = args.files
    if not files:
        files = capture(["dasgoclient", "-query", f"file dataset={args.dataset}"])
        if not files:
            raise AgentError("DAS returned no files for this dataset")
        try:
            sites = capture(["dasgoclient", "-query", f"site dataset={args.dataset}"])
            log("sites: " + ", ".join(sites))
            if sites and all(("Tape" in s or "MSS" in s) for s in sites):
                log("WARNING: dataset seems to be on tape only; xrootd reads will likely fail.")
        except AgentError as e:
            log(f"site lookup failed: {e}")
    files = files[:args.nfiles]
    plan["input_files"] = files
    with open(os.path.join(logs, "input_files.txt"), "w") as f:
        f.write("\n".join(files) + "\n")

    def url(lfn):
        return "root://cms-xrd-global.cern.ch/" + lfn if lfn.startswith("/store/") else lfn

    # 3. does the first file have what we need?
    if not args.skip_content_check:
        code, _ = run(in_era(era, ["bash", os.path.join(CMSSW_DIR, "check_content.sh"), area,
                                   url(files[0]), os.path.join(logs, "eventcontent.txt")]),
                      check=False)
        if code == 3:
            raise AgentError("Input is MiniAOD; use the AOD/AODSIM dataset.")
        if code != 0:
            raise AgentError("Required collections missing (or file unreadable). "
                             f"See {logs}/eventcontent.txt; rerun with --skip-content-check to force.")

    # 4. cmsRun into a local staging dir, then copy to the output (EOS)
    stage = tempfile.mkdtemp(prefix=f"pca_{plan['name']}_", dir=os.environ.get("TMPDIR", "/tmp"))
    cms_args = [f"inputFiles_load={os.path.join(logs, 'input_files.txt')}",
                f"globalTag={plan['globaltag']}", f"outputDir={stage}", f"outputName={plan['name']}",
                f"maxEvents={args.nevents}", f"eventsPerIg={args.per_ig}"]
    if args.no_pt_cut:
        cms_args += ["trackPtMin=0", "muonPtMin=0"]
    run(in_era(era, ["bash", os.path.join(CMSSW_DIR, "run_cmsrun.sh"), area,
                     os.path.join(CMSSW_DIR, "ispy_aod_cfg.py")] + cms_args),
        log_path=os.path.join(logs, "cmsRun.log"))

    ig_dir = os.path.join(out, "ig")
    produced = sorted(glob.glob(os.path.join(stage, "*.ig")))
    if not produced:
        raise AgentError(f"cmsRun produced no ig files (see {logs}/cmsRun.log)")
    for p in produced:
        shutil.copy2(p, ig_dir)
    shutil.rmtree(stage, ignore_errors=True)
    log(f"{len(produced)} ig files -> {ig_dir}")


def event_igs(ig_dir, name):
    """Event ig files (excluding the *_ES.ig geometry file), in numeric order."""
    paths = [p for p in glob.glob(os.path.join(ig_dir, f"{name}_*.ig")) if not p.endswith("_ES.ig")]

    def index(p):
        tail = os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0
    return sorted(paths, key=index)


def step_check_ig(out, igs):
    code, _ = run([sys.executable, os.path.join(HERE, "agent", "check_ig.py")] + igs,
                  log_path=os.path.join(out, "logs", "check_ig.txt"), check=False)
    if code != 0:
        raise AgentError("ig files contain no tracks — conversion failed. "
                         f"See {out}/logs/check_ig.txt and cmsRun.log")


# ════════════════════════════════════════════════════════
#  step 2: ig -> npz + html
# ════════════════════════════════════════════════════════

def step_pointcloud(out, igs, subtitle):
    try:
        from agent import pointcloud, visualize
    except ImportError as e:
        raise AgentError(f"{e}. Install once with:  python3 -m pip install --user numpy plotly")

    events_dir = os.path.join(out, "ig", "Events")
    summary = []
    for ig in igs:
        with zipfile.ZipFile(ig) as z:
            for entry in z.namelist():
                if "Events/" not in entry or entry.endswith("/"):
                    continue
                rel = entry[entry.index("Events/") + len("Events/"):]     # Run_X/Event_Y
                raw = z.read(entry)
                dst = os.path.join(events_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(raw)

                data = pointcloud.load_ig_text(raw.decode("utf-8", errors="replace"))
                run_no, ev_no = pointcloud.event_id(data)
                if run_no is None:
                    run_part, ev_part = rel.split("/")[:2]
                    run_no, ev_no = int(run_part.split("_")[1]), int(ev_part.split("_")[1])

                arrays = pointcloud.build(data)
                counts = pointcloud.object_counts(arrays)

                npz = os.path.join(out, "pointcloud", f"Run_{run_no}", f"Event_{ev_no}.npz")
                os.makedirs(os.path.dirname(npz), exist_ok=True)
                pointcloud.save_npz(npz, arrays, run_no, ev_no)

                page = os.path.join(out, "visualize", f"Run_{run_no}", f"Event_{ev_no}.html")
                os.makedirs(os.path.dirname(page), exist_ok=True)
                with open(page, "w", encoding="utf-8") as f:
                    f.write(visualize.render_page(arrays, counts, run_no, ev_no, subtitle))

                summary.append({"run": run_no, "event": ev_no, "ig": os.path.basename(ig),
                                "counts": counts})
                log(f"Run {run_no} Event {ev_no}: tracks={counts['tracks']} "
                    f"electrons={counts['elec_tracks']} pv={counts['vertices']} "
                    f"EB={counts['ecal_eb']} EE={counts['ecal_ee']} HB={counts['hcal_hb']}")
    if not summary:
        raise AgentError("No events found inside the ig files")
    return summary


# ════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", help="DAS dataset name /Primary/Processed/AODSIM")
    p.add_argument("--ig", nargs="+", help="Existing ig files: skip CMSSW, only make npz + html")
    p.add_argument("--name", help="Output name (default: <primary>__<campaign>)")
    p.add_argument("--out", default=user_default_dir("PointCloud_Agent_output", "PointCloud_Agent_output"),
                   help="Output base directory (default: EOS user area on lxplus, else ~)")
    p.add_argument("--cmssw-base", default=cmssw_default_dir(),
                   help="Where CMSSW areas are created (default: AFS work area on lxplus, else ~)")
    p.add_argument("--nevents", type=int, default=100, help="Events to process (default 100, -1 = all)")
    p.add_argument("--per-ig", type=int, default=10, help="Events per ig file (default 10)")
    p.add_argument("--nfiles", type=int, default=1, help="Input files from DAS to use (default 1)")
    p.add_argument("--files", nargs="+", help="Explicit input LFNs/URLs instead of the DAS file list")
    p.add_argument("--globaltag", help="Override the GlobalTag")
    p.add_argument("--release", help="Override the DAS release used to pick the era, e.g. CMSSW_12_0_2")
    p.add_argument("--no-pt-cut", action="store_true", help="Disable ISpy track/muon pT cuts")
    p.add_argument("--skip-content-check", action="store_true", help="Do not check collections first")
    p.add_argument("--steps", default="ig,pointcloud",
                   help="Comma list of steps: ig, pointcloud (default both). "
                        "'pointcloud' alone reuses ig files already in <out>/<name>/ig")
    p.add_argument("--dry-run", action="store_true", help="Only print the plan")
    args = p.parse_args(argv)
    if not args.dataset and not args.ig:
        p.error("give a DAS dataset name or --ig files")
    return args


def main(argv=None):
    args = parse_args(argv)
    steps = set(s.strip() for s in args.steps.split(","))
    try:
        plan = make_plan(args)
        out = os.path.abspath(os.path.join(args.out, plan["name"]))
        plan["output"] = out
        log("plan:\n" + json.dumps(plan, indent=2))
        if args.dry_run:
            return 0

        for sub in ("ig", "logs", "pointcloud", "visualize"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)

        if args.ig:
            igs = []
            for p in args.ig:
                dst = os.path.join(out, "ig", os.path.basename(p))
                if not os.path.exists(dst) or not os.path.samefile(p, dst):
                    shutil.copy2(p, dst)
                igs.append(dst)
        else:
            if "ig" in steps:
                step_ig(args, plan, out)
            igs = event_igs(os.path.join(out, "ig"), plan["name"])
            if not igs:
                raise AgentError(f"No ig files in {out}/ig")
        step_check_ig(out, igs)

        summary = []
        if "pointcloud" in steps:
            subtitle = args.dataset or ", ".join(os.path.basename(p) for p in args.ig)
            summary = step_pointcloud(out, igs, subtitle)

        manifest = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": os.uname().nodename,
            "agent_git": git_info(),
            "command": sys.argv,
            "plan": plan,
            "options": {"nevents": args.nevents, "per_ig": args.per_ig, "nfiles": args.nfiles,
                        "no_pt_cut": args.no_pt_cut},
            "events": summary,
        }
        with open(os.path.join(out, "logs", "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"done: {len(summary)} events -> {out}")
        return 0
    except (AgentError, ValueError) as e:
        log(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
