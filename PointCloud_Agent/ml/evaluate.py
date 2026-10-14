"""
Apply a trained run (ml/train.py) to any pack, e.g. a sample the model never saw (DY -> ee,
DYJetsToLL, ttbar). The model uses its own input config and normalisation from the
checkpoint, never the new pack's.

Working points come from the training pack's val split (the run's own events): the score
cuts that keep 90 / 95 / 99 % (--efficiencies) of the true muon tracks with pT > pt_min. On
the target pack it then reports, per event category and per working point,
    muon efficiency    muon-labelled tracks above the cut
    electron fake rate tracks matched to a GSF electron (and not to a muon) above the cut
    other fake rate    remaining tracks (hadrons, pileup) above the cut
for tracks with pT > pt_min. Event categories from the reconstructed leptons above pt_min
(used to split the result only, never as input): mumu (>= 2 muons, no electron),
ee (>= 2 electrons, no muon), emu (>= 1 of each), other. A track counts as an electron
when its inner direction is within --e-angle (rad) and its inner point within --e-d0 (m)
of a GSF electron with pT > pt_min; these cuts are provisional, check them with the
"electron match" lines on the real sample. Packs made before the electron arrays have only
the "all" category and no electron rates.

    python3 -m ml.evaluate --run /eos/user/j/jieun/ml/runs/B_conv \\
        --pack /eos/user/j/jieun/ml/packed/dyll_ul16_v1 --split test \\
        --ref-pack /eos/user/j/jieun/ml/packed/zmm_ul16_v1     # B_conv / B_mlp: see --ref-pack
    python3 -m ml.evaluate --run RUN --pack PACK --out OUT --efficiencies 0.9 0.99

Writes <out>/eval_metrics.json and <out>/eval_predictions.npz (per track: score, label, pt,
eta, event, category, is_electron, e_angle, e_d0). Default out: <run>/eval/<pack>_<split>.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import packed  # noqa: E402
from ml.metrics import pass_rate, run_pack, summarise, threshold_at_efficiency  # noqa: E402
from ml.particlenet import TrackParticleNet  # noqa: E402
from ml.train import EventDataset, collate, run_epoch  # noqa: E402

CATEGORIES = ("mumu", "ee", "emu", "other")


def event_category(raw, pt_min):
    n_mu = int((raw["glb_pt"] > pt_min).sum())
    n_e = int((raw["ele_pt"] > pt_min).sum())
    if n_mu >= 2 and not n_e:
        return "mumu"
    if n_e >= 2 and not n_mu:
        return "ee"
    if n_mu and n_e:
        return "emu"
    return "other"


def predict(model, split, cfg, norm, device, args):
    loader = torch.utils.data.DataLoader(EventDataset(split, cfg, norm), batch_size=args.batch_size,
                                         collate_fn=collate, num_workers=args.workers)
    return run_epoch(model, loader, device, "track", 1.0, amp=args.amp and device.type == "cuda",
                     limit=args.limit_batches, collect=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder of ml/train.py")
    p.add_argument("--checkpoint", default="best.pt", help="file in the run folder (default best.pt)")
    p.add_argument("--pack", required=True, help="pack to evaluate on (ml/pack.py)")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--out", help="output folder (default <run>/eval/<pack name>_<split>)")
    p.add_argument("--ref-pack", help="the run's training pack, for the working points (default: from "
                   "its config.json; needed for runs whose config has a condor scratch path)")
    p.add_argument("--efficiencies", type=float, nargs="+", default=[0.90, 0.95, 0.99],
                   help="muon efficiencies of the working points (default 0.90 0.95 0.99)")
    p.add_argument("--e-angle", type=float, default=0.01, help="electron match: inner direction angle in rad")
    p.add_argument("--e-d0", type=float, default=0.005, help="electron match: inner point distance in m")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit-batches", type=int, default=None, help="quick tests only")
    args = p.parse_args(argv)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    ck = torch.load(os.path.join(args.run, args.checkpoint), map_location=device, weights_only=False)
    config = ck["config"]
    if config["args"]["task"] != "track":
        raise SystemExit("ml/evaluate.py handles --task track runs only")
    cfg = packed.InputConfig(**config["input_config"])
    norm = packed.Normalizer.from_json(config["normalisation"], cfg)
    model = TrackParticleNet(**config["model"]).to(device)
    model.load_state_dict(ck["model"])
    pt_min = cfg.pt_min
    out = args.out or os.path.join(args.run, "eval", f"{os.path.basename(os.path.normpath(args.pack))}_{args.split}")
    os.makedirs(out, exist_ok=True)

    # working points on the run's own val split
    ref_pack = run_pack(config, args.ref_pack)
    ref = predict(model, packed.PackedSplit(os.path.join(ref_pack, "val")), cfg, norm, device, args)
    hi = ref["pt"] > pt_min
    cuts = {f"{e:g}": threshold_at_efficiency(ref["score"][hi], ref["label"][hi], e) for e in args.efficiencies}

    # target
    split = packed.PackedSplit(os.path.join(args.pack, args.split))
    res = predict(model, split, cfg, norm, device, args)
    n = len(res["score"])
    # per-track electron info from the pack, in prediction order (events in loader order,
    # tracks in pack order; events without tracks have no rows)
    _, first = np.unique(res["event"], return_index=True)
    events = res["event"][np.sort(first)].astype(int)
    cat_of, e_angle, e_d0, e_pt = {}, [], [], []
    for e in events:
        raw = split.event(int(e))
        cat_of[int(e)] = event_category(raw, pt_min) if split.has_electrons else "other"
        e_angle.append(raw["track_e_angle"])
        e_d0.append(raw["track_e_d0"])
        e_pt.append(raw["track_e_pt"])
    e_angle, e_d0, e_pt = (np.concatenate(x)[:n] for x in (e_angle, e_d0, e_pt))
    category = np.array([cat_of[int(e)] for e in res["event"]])
    label = res["label"] > 0.5
    is_e = (~label) & (e_angle < args.e_angle) & (e_d0 < args.e_d0) & (e_pt > pt_min)
    hi = res["pt"] > pt_min

    metrics = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "run": os.path.abspath(args.run), "checkpoint": args.checkpoint,
        "pack": os.path.abspath(args.pack), "split": args.split, "ref_pack": ref_pack,
        "pt_min": pt_min, "e_angle": args.e_angle, "e_d0": args.e_d0,
        "has_electron_info": split.has_electrons,
        "working_points": cuts,
        "all": summarise(res["score"], res["label"], res["pt"], pt_min, "track"),
        "categories": {},
    }
    groups = (("all",) + CATEGORIES) if split.has_electrons else ("all",)
    for g in groups:
        sel = hi & (category == g if g != "all" else True)
        row = {"events": int(len(set(res["event"][category == g]))) if g != "all" else int(len(events)),
               "tracks": int(sel.sum()), "muons": int((sel & label).sum()),
               "electrons": int((sel & is_e).sum()), "others": int((sel & ~label & ~is_e).sum()),
               "working_points": {}}
        for e, cut in cuts.items():
            if cut is None:
                continue
            row["working_points"][e] = {
                "muon_efficiency": pass_rate(res["score"], sel & label, cut),
                "electron_fake_rate": pass_rate(res["score"], sel & is_e, cut),
                "other_fake_rate": pass_rate(res["score"], sel & ~label & ~is_e, cut),
            }
        metrics["categories"][g] = row
    if split.has_electrons:
        # how well the provisional electron cuts separate: tracks above pt_min in ee events
        ee = hi & (category == "ee") & ~label & (e_pt > pt_min)
        a = e_angle[ee & np.isfinite(e_angle)]
        metrics["electron_match"] = {
            "tracks_in_ee_events": int(ee.sum()),
            "angle_quantiles": {q: float(np.quantile(a, float(q))) for q in ("0.1", "0.5", "0.9")} if len(a) else None,
            "angle_below": {f"{c:g}": int((a < c).sum()) for c in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)},
        }
    with open(os.path.join(out, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(out, "eval_predictions.npz"), score=res["score"], label=res["label"],
             pt=res["pt"], eta=res["eta"], event=res["event"], category=category, is_electron=is_e,
             e_angle=e_angle, e_d0=e_d0)

    print(f"{args.pack} [{args.split}]: {len(events)} events, {int(hi.sum())} tracks with pT > {pt_min:g}")
    print(f"working points from {ref_pack}/val: " + ", ".join(f"eff {e} -> cut {c:.4g}" for e, c in cuts.items() if c is not None))
    a = metrics["all"]
    print(f"auc_pt {a.get('auc_pt')}  ap_pt {a.get('ap_pt')}")
    for g, row in metrics["categories"].items():
        print(f"[{g}] events {row['events']}, tracks {row['tracks']} (muons {row['muons']}, "
              f"electrons {row['electrons']}, others {row['others']})")
        for e, wp in row["working_points"].items():
            fmt = (lambda r: "  -   " if r["rate"] is None else f"{r['rate']:.4f}")
            print(f"   eff {e}: muon eff {fmt(wp['muon_efficiency'])}  e fake {fmt(wp['electron_fake_rate'])}"
                  f"  other fake {fmt(wp['other_fake_rate'])}")
    if "electron_match" in metrics:
        print(f"electron match (ee events): {metrics['electron_match']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
