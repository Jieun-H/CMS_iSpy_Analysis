#!/usr/bin/env python3
"""
PointCloud_Agent: CMS AOD/AODSIM dataset -> iSpy ig -> point cloud (npz) + visualization (html).

Run on lxplus, normally from inside the CMSSW area it lives in (CMSSW_X/src/CMS_iSpy_Analysis/PointCloud_Agent),
either inside the matching cmssw-elN container or from the el9 host shell (the agent then wraps CMSSW
commands in cmssw-elN itself):

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


def find_cmssw_area(start):
    """CMSSW area containing `start` (a directory with .SCRAM/<arch>), or None."""
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        scram = os.path.join(d, ".SCRAM")
        if os.path.isdir(scram) and os.path.basename(d).startswith("CMSSW_"):
            archs = [a for a in os.listdir(scram) if os.path.isdir(os.path.join(scram, a))
                     and eras.arch_os(a) is not None]
            return {"path": d, "release": os.path.basename(d), "arch": archs[0] if archs else None}
        d = os.path.dirname(d)
    return None


def host_os():
    """Major RHEL-family version of the OS we are running on (inside a container: the container's)."""
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("VERSION_ID="):
                    return int(line.split("=", 1)[1].strip().strip('"').split(".")[0])
    except (OSError, ValueError):
        pass
    try:
        with open("/etc/redhat-release") as f:
            for tok in f.read().split():
                if tok[0].isdigit():
                    return int(tok.split(".")[0])
    except (OSError, ValueError):
        pass
    return None


def in_os(scram_arch, cmd):
    """Run cmd directly if we are already on the OS the arch needs, else via cmssw-elN."""
    need, have = eras.arch_os(scram_arch), host_os()
    if need is None or need == have:
        return cmd
    if os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"):
        raise AgentError(f"{scram_arch} needs el{need} but this shell is an el{have} container. "
                         f"Exit it and run 'cmssw-el{need}' (or run from the plain lxplus shell).")
    name = f"cmssw-el{need}"
    wrapper = shutil.which(name) or f"/cvmfs/cms.cern.ch/common/{name}"
    if not os.path.exists(wrapper):
        raise AgentError(f"{name} not found — is this lxplus with /cvmfs/cms.cern.ch?")
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

    # Which CMSSW area: the one this code lives in (CMSSW_X/src/...), unless --cmssw-base is given
    enclosing = None if args.cmssw_base else find_cmssw_area(HERE)
    if enclosing:
        if not era.matches(enclosing["release"]):
            raise AgentError(
                f"This code is inside {enclosing['release']}, but the dataset was produced with {release} "
                f"(era '{era.key}' needs {' / '.join(p + 'X' for p in era.release_prefixes)}, "
                f"e.g. {era.cmssw}).\n"
                f"  Set up {era.cmssw} (SCRAM_ARCH={era.scram_arch}) and clone the repository into its src/, "
                f"or pass --cmssw-base DIR to let the agent create one.")
        area, cmssw, arch = enclosing["path"], enclosing["release"], enclosing["arch"] or era.scram_arch
    else:
        base = os.path.abspath(args.cmssw_base or cmssw_default_dir())
        area, cmssw, arch = os.path.join(base, era.cmssw), era.cmssw, era.scram_arch

    plan.update({
        "das_release": release, "era": era.key, "cmssw": cmssw, "scram_arch": arch,
        "cmssw_area": area, "cmssw_area_source": "enclosing" if enclosing else "agent",
        "ispy_branch": era.ispy_branch, "services_branch": era.services_branch,
        "globaltag": gt, "era_validated": era.validated,
    })
    return plan


# ════════════════════════════════════════════════════════
#  step 1: AOD -> ig (lxplus)
# ════════════════════════════════════════════════════════

def step_ig(args, plan, out):
    era = eras.era_for_release(plan["das_release"])
    if not era.validated:
        log(f"NOTE: era '{era.key}' has not been validated end-to-end yet. {era.notes}")

    try:
        proxy_ok = subprocess.run(["voms-proxy-info", "--exists", "--valid", "0:30"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        proxy_ok = True   # no voms client in this shell; xrootd will complain if the proxy is missing
        log("WARNING: voms-proxy-info not found, cannot check the grid proxy")
    if not proxy_ok:
        raise AgentError("No valid grid proxy (or <30 min left). Run:\n"
                         "  voms-proxy-init --rfc --voms cms --valid 168:00")

    logs = os.path.join(out, "logs")
    area, arch = plan["cmssw_area"], plan["scram_arch"]

    def in_era(cmd):
        return in_os(arch, cmd)

    # 1. CMSSW area + ISpy build (first run per era takes 10-20 min; later only a quick scram b)
    log(f"CMSSW area: {area}  ({plan['cmssw_area_source']}, ISpy {era.ispy_branch})")
    run(in_era(["bash", os.path.join(CMSSW_DIR, "setup_cmssw.sh"), os.path.dirname(area),
                os.path.basename(area), arch, era.ispy_branch, era.services_branch]),
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
    files = files[args.first_file:args.first_file + args.nfiles]
    if not files:
        raise AgentError(f"No input files left after --first-file {args.first_file}")
    plan["input_files"] = files
    with open(os.path.join(logs, "input_files.txt"), "w") as f:
        f.write("\n".join(files) + "\n")

    def url(lfn):
        return "root://cms-xrd-global.cern.ch/" + lfn if lfn.startswith("/store/") else lfn

    # 3. does the first file have what we need?
    if not args.skip_content_check:
        code, _ = run(in_era(["bash", os.path.join(CMSSW_DIR, "check_content.sh"), area,
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
    run(in_era(["bash", os.path.join(CMSSW_DIR, "run_cmsrun.sh"), area,
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

def step_pointcloud(out, igs, subtitle, html=True):
    try:
        from agent import pointcloud
        if html:
            from agent import visualize   # plotly is only needed for the HTML pages
    except ImportError as e:
        raise AgentError(f"{e}. Install once with:  python3 -m pip install --user numpy plotly")

    events_dir = os.path.join(out, "ig", "Events")
    summary = []
    seen = set()
    for ig in igs:
        with zipfile.ZipFile(ig) as z:
            for entry in z.namelist():
                if "Events/" not in entry or entry.endswith("/"):
                    continue
                rel = entry[entry.index("Events/") + len("Events/"):]     # Run_X/Event_Y
                raw = z.read(entry)

                data = pointcloud.load_ig_text(raw.decode("utf-8", errors="replace"))
                run_no, ev_no, ls_no = pointcloud.event_id(data)
                if run_no is None:
                    run_part, ev_part = rel.split("/")[:2]
                    run_no, ev_no = int(run_part.split("_")[1]), int(ev_part.split("_")[1])
                # (run, lumi, event) is the unique key; MC is mostly run=1 and several input files
                # can repeat (run, event), so lumi goes into the file name.
                stem = f"Event_{ev_no}" + (f"_LS{ls_no}" if ls_no is not None else "")
                key = (run_no, ls_no, ev_no)
                if key in seen:
                    log(f"WARNING: duplicate event run={run_no} lumi={ls_no} event={ev_no} "
                        f"({os.path.basename(ig)}); overwriting the earlier output")
                seen.add(key)

                dst = os.path.join(events_dir, f"Run_{run_no}", stem)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(raw)

                arrays = pointcloud.build(data)
                counts = pointcloud.object_counts(arrays)

                npz = os.path.join(out, "pointcloud", f"Run_{run_no}", f"{stem}.npz")
                os.makedirs(os.path.dirname(npz), exist_ok=True)
                pointcloud.save_npz(npz, arrays, run_no, ev_no, ls_no)

                if html:
                    page = os.path.join(out, "visualize", f"Run_{run_no}", f"{stem}.html")
                    os.makedirs(os.path.dirname(page), exist_ok=True)
                    with open(page, "w", encoding="utf-8") as f:
                        f.write(visualize.render_page(arrays, counts, run_no, ev_no, subtitle))

                summary.append({"run": run_no, "lumi": ls_no, "event": ev_no, "ig": os.path.basename(ig),
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
    p.add_argument("--name", help="Output name (default: <primary>__<processed>)")
    p.add_argument("--out", default=user_default_dir("PointCloud_Agent_output", "PointCloud_Agent_output"),
                   help="Output base directory (default: EOS user area on lxplus, else ~)")
    p.add_argument("--cmssw-base",
                   help="Create/use CMSSW areas under this directory instead of the CMSSW area this code "
                        "lives in (default when not inside one: AFS work area on lxplus, else ~/pca_cmssw)")
    p.add_argument("--nevents", type=int, default=100, help="Events to process (default 100, -1 = all)")
    p.add_argument("--per-ig", type=int, default=10, help="Events per ig file (default 10)")
    p.add_argument("--nfiles", type=int, default=1, help="Input files from DAS to use (default 1)")
    p.add_argument("--first-file", type=int, default=0,
                   help="Skip this many files of the DAS list (next batch of a dataset; use a new --name)")
    p.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                   help="Write the per-event HTML pages (default); --no-html skips them")
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
            summary = step_pointcloud(out, igs, subtitle, html=args.html)

        manifest = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": os.uname().nodename,
            "agent_git": git_info(),
            "command": sys.argv,
            "plan": plan,
            "options": {"nevents": args.nevents, "per_ig": args.per_ig, "nfiles": args.nfiles,
                        "first_file": args.first_file, "no_pt_cut": args.no_pt_cut,
                        "html": args.html},
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
