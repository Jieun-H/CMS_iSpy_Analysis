"""
Figures and tables for a presentation, from finished runs (ml/train.py), the single-variable
baselines (ml/baseline.py, recomputed here on the same split) and, optionally, evaluations
on other samples (ml/evaluate.py). numpy + matplotlib, no torch, no GPU.

Everything is for tracks with pT > pt_min (the tracks a muon label can exist for).

    roc.png             muon efficiency vs non-muon pass rate (log), models and baselines
    learning_curves.png train / val loss and val auc_pt per epoch
    scores.png          model score for muon and non-muon tracks
    vs_pt.png, vs_eta.png  muon efficiency and non-muon pass rate per bin at one working point
    summary.md / .json  auc_pt (+- bootstrap over events), ap_pt, pass rate at fixed efficiencies
    evals.md            per sample / event category / working point (with --evals)

    # lxplus: matplotlib comes with the LCG view
    source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh
    python3 -m ml.report --runs /eos/user/j/jieun/ml/runs/B_conv /eos/user/j/jieun/ml/runs/B_mlp \\
        --out /eos/user/j/jieun/ml/reports/zmm_v1
    python3 -m ml.report --runs Conv=RUN_A MLP=RUN_B --out OUT --baselines pt iso_rel n_seg \\
        --evals RUN_A/eval/dyll_ul16_v1_test --wp 0.99 --formats png pdf
"""
import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import baseline, packed  # noqa: E402
from ml.metrics import (average_precision, bootstrap, pass_rate, roc_auc, roc_curve,  # noqa: E402
                        run_pack, threshold_at_efficiency)

# reference palette, light mode (dataviz skill): categorical slots in fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PT_BINS = (10, 15, 20, 30, 40, 50, 70, 100, 1e9)
ETA_BINS = (0, 0.9, 1.2, 2.1, 2.4, 2.5)         # DT barrel / overlap / CSC endcap / edge


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "font.size": 11, "legend.frameon": False,
        "axes.titlesize": 12, "axes.titleweight": "bold",
    })


def _save(fig, out, name, formats):
    for f in formats:
        fig.savefig(os.path.join(out, f"{name}.{f}"), dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------- inputs

def load_run(spec):
    name, _, path = spec.rpartition("=") if "=" in spec else (None, None, spec)
    path = os.path.normpath(path)
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(path, "metrics.jsonl")) as f:
        history = [json.loads(line) for line in f if line.strip()]
    pred = dict(np.load(os.path.join(path, "test_predictions.npz")))
    return {"name": name or os.path.basename(path), "path": path, "config": config,
            "history": history, "pred": pred}


def fill_eta(pred, split):
    """Old test_predictions.npz have no eta: take it from the pack (prediction order =
    events in loader order, tracks in pack order)."""
    if "eta" in pred:
        return pred
    _, first = np.unique(pred["event"], return_index=True)
    events = pred["event"][np.sort(first)].astype(int)
    eta = np.concatenate([split.event(e)["track_info"][:, 1] for e in events])
    if len(eta) < len(pred["score"]):
        raise SystemExit("cannot match predictions to the pack (different pack or split?)")
    pred["eta"] = eta[:len(pred["score"])]
    return pred


def baselines(split, names, pt_min, match_tol, cones):
    args = argparse.Namespace(pt_min=pt_min, match_tol=match_tol, **cones)
    labels, values, events = [], [], []
    for e in range(len(split)):
        lab, val = baseline.event_variables(split.event(e), args)
        labels.append(lab)
        values.append(val)
        events.append(np.full(len(lab), e))
    label = np.concatenate(labels)
    value = np.concatenate(values)
    out = {}
    for v in names:
        x = value[:, baseline.VARIABLES.index(v)]
        sign = 1 if (roc_auc(x, label) or 0.5) >= 0.5 else -1
        out[v] = {"score": sign * x, "label": label, "event": np.concatenate(events), "sign": sign}
    return out


# ---------------------------------------------------------------- numbers

def score_summary(score, label, event, efficiencies, n_boot):
    label = np.asarray(label).astype(bool)
    row = {"tracks": int(len(label)), "muons": int(label.sum()),
           "auc_pt": roc_auc(score, label), "ap_pt": average_precision(score, label),
           "auc_pt_err": bootstrap(lambda i: roc_auc(score[i], label[i]), event, n_boot) if n_boot else None,
           "non_muon_pass": {}}
    for e in efficiencies:
        cut = threshold_at_efficiency(score, label, e)
        row["non_muon_pass"][f"{e:g}"] = None if cut is None else pass_rate(score, ~label, cut)["rate"]
    return row


def _fmt(x, digits=4):
    return "-" if x is None else f"{x:.{digits}f}"


def write_summary(out, rows, efficiencies, meta):
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    head = ["", "kind", "auc_pt", "ap_pt"] + [f"non-muon pass @ eff {e:g}" for e in efficiencies]
    lines = [f"Tracks with pT > {meta['pt_min']:g} GeV in {meta['pack']} [{meta['split']}]: "
             f"{meta['tracks']} tracks, {meta['muons']} muons "
             f"(muon fraction {meta['muons'] / max(meta['tracks'], 1):.3f} = ap_pt of a random score).",
             "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in rows.items():
        auc = _fmt(r["auc_pt"]) + (f" ± {r['auc_pt_err']:.4f}" if r["auc_pt_err"] else "")
        lines.append("| " + " | ".join([name, r["kind"], auc, _fmt(r["ap_pt"])]
                                       + [_fmt(r["non_muon_pass"][f"{e:g}"]) for e in efficiencies]) + " |")
    lines += ["", "auc_pt error: std over bootstrap resamplings of whole events.",
              "non-muon pass: fraction of non-muon tracks above the cut that keeps that muon efficiency "
              "(cut set on this split itself)."]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_evals(out, evals):
    lines = ["| sample | category | events | muons | electrons | others | WP eff | muon eff | e pass | other pass |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for path in evals:
        with open(os.path.join(path, "eval_metrics.json")) as f:
            m = json.load(f)
        sample = f"{os.path.basename(m['pack'])} [{m['split']}]"
        for cat, row in m["categories"].items():
            for e, wp in row["working_points"].items():
                lines.append("| " + " | ".join([
                    sample, cat, str(row["events"]), str(row["muons"]), str(row["electrons"]),
                    str(row["others"]), e, _fmt(wp["muon_efficiency"]["rate"]),
                    _fmt(wp["electron_fake_rate"]["rate"]), _fmt(wp["other_fake_rate"]["rate"])]) + " |")
    lines += ["", "Working points: score cuts keeping that muon efficiency on the training pack's val split. "
              "e pass / other pass: fraction of electron / other non-muon tracks above the cut."]
    with open(os.path.join(out, "evals.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- figures

def fig_roc(plt, curves, out, formats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (lo, title) in zip(axes, ((0.0, "All efficiencies"), (0.9, "High-efficiency region"))):
        for i, (name, c) in enumerate(curves.items()):
            tpr, fpr, _ = roc_curve(c["score"], c["label"])
            keep = (fpr > 0) & (tpr >= lo)
            ax.plot(tpr[keep], fpr[keep], color=SERIES[i % len(SERIES)],
                    linestyle="-" if c["kind"] == "model" else "--", label=name)
        ax.set_yscale("log")
        ax.set_xlim(lo, 1.0)
        ax.set_xlabel("Muon track efficiency")
        ax.set_ylabel("Non-muon tracks passing")
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.02), title="solid: model, dashed: single variable")
    fig.tight_layout(rect=(0, 0.06 + 0.05 * ((len(labels) - 1) // 4), 1, 1))
    _save(fig, out, "roc", formats)
    plt.close(fig)


def fig_learning(plt, runs, out, formats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, r in enumerate(runs):
        ep = [h["epoch"] for h in r["history"]]
        col = SERIES[i % len(SERIES)]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], color=col, linestyle="--",
                     label=f"{r['name']} train")
        axes[0].plot(ep, [h["val_loss"] for h in r["history"]], color=col, label=f"{r['name']} val")
        axes[1].plot(ep, [h.get("val_auc_pt") for h in r["history"]], color=col, marker="o",
                     markersize=4, label=r["name"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss per track")
    axes[0].set_title("Loss", loc="left")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUC (pT > pt_min)")
    axes[1].set_title("Validation AUC", loc="left")
    axes[1].legend()
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    _save(fig, out, "learning_curves", formats)
    plt.close(fig)


def fig_scores(plt, runs, sel, out, formats):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
    bins = np.linspace(0, 1, 51)
    for ax, r in zip(axes[0], runs):
        p = r["pred"]
        s, lab = p["score"][sel[r["name"]]], p["label"][sel[r["name"]]] > 0.5
        ax.hist(s[lab], bins=bins, histtype="step", linewidth=1.6, color=SERIES[0], label="muon tracks")
        ax.hist(s[~lab], bins=bins, histtype="step", linewidth=1.6, color=SERIES[1], label="non-muon tracks")
        ax.set_yscale("log")
        ax.set_xlabel("Model score")
        ax.set_ylabel("Tracks")
        ax.set_title(r["name"], loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out, "scores", formats)
    plt.close(fig)


def _binned(x, sel_pos, sel_neg, passed, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (x >= lo) & (x < hi)
        out = []
        for sel in (sel_pos, sel_neg):
            n = int((b & sel).sum())
            k = int((b & sel & passed).sum())
            r = k / n if n else np.nan
            out.append((r, np.sqrt(r * (1 - r) / n) if n else np.nan, n))
        rows.append(out)
    return rows


def fig_binned(plt, runs, sel, wp, out, formats, var, edges, xlabel, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    centres = np.arange(len(edges) - 1)
    ticks = [f"{lo:g}–{hi:g}" if hi < 1e8 else f">{lo:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    for i, r in enumerate(runs):
        p = r["pred"]
        s = sel[r["name"]]
        lab = p["label"] > 0.5
        cut = threshold_at_efficiency(p["score"][s], lab[s], wp)
        x = np.abs(p[var]) if var == "eta" else p[var]
        rows = _binned(x, s & lab, s & ~lab, p["score"] >= cut, edges)
        off = (i - (len(runs) - 1) / 2) * 0.12
        for ax, k in zip(axes, (0, 1)):
            y = np.array([row[k][0] for row in rows])
            err = np.array([row[k][1] for row in rows])
            if k == 1:                       # log axis: a zero rate has no point
                y = np.where(y > 0, y, np.nan)
            low = np.minimum(err, 0.9 * y) if k == 1 else err      # keep the log axis finite
            ax.errorbar(centres + off, y, yerr=[low, err], color=SERIES[i % len(SERIES)], marker="o",
                        markersize=5, linewidth=1.6, capsize=2, label=r["name"])
    axes[0].set_ylabel("Muon track efficiency")
    axes[0].set_title(f"Muon efficiency (cut for {wp:.0%} overall)", loc="left")
    axes[1].set_ylabel("Non-muon tracks passing")
    axes[1].set_yscale("log")
    axes[1].set_title("Non-muon pass rate", loc="left")
    for ax in axes:
        ax.set_xticks(centres)
        ax.set_xticklabels(ticks, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlabel(xlabel)
        ax.legend()
    fig.tight_layout()
    _save(fig, out, name, formats)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders, optionally NAME=FOLDER")
    p.add_argument("--out", required=True, help="folder for figures and tables")
    p.add_argument("--pack", help="pack of the runs' test split (default: from the first run's config)")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--baselines", nargs="*", default=["pt", "iso_rel", "n_seg", "n_rpc", "ecal_e"],
                   choices=baseline.VARIABLES, help="single-variable baselines to show")
    p.add_argument("--efficiencies", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    p.add_argument("--wp", type=float, default=0.99, help="muon efficiency for vs_pt / vs_eta (default 0.99)")
    p.add_argument("--bootstrap", type=int, default=200, help="event resamplings for errors (0: none)")
    p.add_argument("--iso-cone", type=float, default=0.3)
    p.add_argument("--muon-cone", type=float, default=0.3)
    p.add_argument("--ecal-cone", type=float, default=0.1)
    p.add_argument("--evals", nargs="*", default=[], help="ml/evaluate.py output folders")
    p.add_argument("--formats", nargs="+", default=["png"], help="png, pdf, svg ...")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(args.out, exist_ok=True)

    runs = [load_run(s) for s in args.runs]
    cfg0 = runs[0]["config"]["input_config"]
    pt_min, match_tol = cfg0["pt_min"], cfg0["match_tol"]
    for r in runs[1:]:
        c = r["config"]["input_config"]
        if (c["pt_min"], c["match_tol"]) != (pt_min, match_tol):
            raise SystemExit(f"{r['name']}: labels differ (pt_min / match_tol), not comparable")
    pack = run_pack(runs[0]["config"], args.pack)
    split = packed.PackedSplit(os.path.join(pack, args.split))
    for r in runs:
        fill_eta(r["pred"], split)

    sel = {r["name"]: r["pred"]["pt"] > pt_min for r in runs}
    curves, rows = {}, {}
    for r in runs:
        s = sel[r["name"]]
        pr = r["pred"]
        curves[r["name"]] = {"score": pr["score"][s], "label": pr["label"][s] > 0.5, "kind": "model"}
        rows[r["name"]] = dict(kind="model", **score_summary(pr["score"][s], pr["label"][s] > 0.5,
                                                             pr["event"][s], args.efficiencies, args.bootstrap))
    if args.baselines:
        cones = {"iso_cone": args.iso_cone, "muon_cone": args.muon_cone, "ecal_cone": args.ecal_cone}
        base = baselines(split, args.baselines, pt_min, match_tol, cones)
        n_model = rows[runs[0]["name"]]["muons"]
        for v, b in base.items():
            name = f"{v} only"
            curves[name] = {"score": b["score"], "label": b["label"], "kind": "baseline"}
            rows[name] = dict(kind="baseline", **score_summary(b["score"], b["label"], b["event"],
                                                               args.efficiencies, args.bootstrap))
        if int(base[args.baselines[0]]["label"].sum()) != n_model:
            print(f"WARNING: baseline muons {int(base[args.baselines[0]]['label'].sum())} != "
                  f"model muons {n_model} (different split or limit?)", file=sys.stderr)

    first = rows[runs[0]["name"]]
    write_summary(args.out, rows, args.efficiencies,
                  {"pack": pack, "split": args.split, "pt_min": pt_min, "tracks": first["tracks"],
                   "muons": first["muons"], "runs": {r["name"]: r["path"] for r in runs}})
    fig_roc(plt, curves, args.out, args.formats)
    fig_learning(plt, runs, args.out, args.formats)
    fig_scores(plt, runs, sel, args.out, args.formats)
    fig_binned(plt, runs, sel, args.wp, args.out, args.formats, "pt", PT_BINS, "Track pT [GeV]", "vs_pt")
    fig_binned(plt, runs, sel, args.wp, args.out, args.formats, "eta", ETA_BINS, "Track |η|", "vs_eta")
    if args.evals:
        write_evals(args.out, args.evals)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
