"""
How well tracks match GSF electrons in a pack (ml/pack.py), to choose the electron label
cuts --e-angle / --e-d0 of ml/train.py, ml/evaluate.py and ml/baseline.py.

GSF tracks are refitted, so unlike tracker muons their end points do not coincide with the
track's; the inner direction (angle, rad) and inner point (d0, m) do, roughly. For tracks
with pT > pt_min that are not a tracker muon, and whose closest GSF electron has pT >
pt_min, it prints log-binned counts of the angle and, for tracks inside --e-angle, of d0.
A good cut sits in a clear gap between the matched peak and the unmatched continuum (as
for muons: <= 1e-6 m vs >= 0.1 m). Then, with the chosen cuts, the electron label count
against the GSF electrons above pt_min (the electron counterpart of check_dataset's
"tracker muons matched" line): unmatched electrons have no track in `tracks`, and more
than one label track per electron means the cuts are too loose.

numpy only (Python 3.9), no GPU:
    python3 -m ml.electron_match --pack /eos/user/j/jieun/ml/packed/dyll_ul16_v1
    python3 -m ml.electron_match --pack PACK --split val --e-angle 0.005 --e-d0 0.002 --json m.json
"""
import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import inputs, packed  # noqa: E402

ANGLE_EDGES = np.r_[0, np.logspace(-5, 0, 11), np.inf]          # rad
D0_EDGES = np.r_[0, np.logspace(-5, -1, 9), np.inf]             # m


def histogram(x, edges):
    counts, _ = np.histogram(x, bins=edges)
    return [{"low": float(lo), "high": float(hi), "count": int(c)}
            for lo, hi, c in zip(edges[:-1], edges[1:], counts)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, help="folder made by ml/pack.py (with electron arrays)")
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--pt-min", type=float, default=10.0, help="track and electron pT (default 10)")
    p.add_argument("--match-tol", type=float, default=inputs.MATCH_TOL, help="tracker muon match in m")
    p.add_argument("--e-angle", type=float, default=packed.E_ANGLE, help="angle cut to check, rad")
    p.add_argument("--e-d0", type=float, default=packed.E_D0, help="d0 cut to check, m")
    p.add_argument("--json", help="also write the result to this file")
    args = p.parse_args(argv)

    split = packed.PackedSplit(os.path.join(args.pack, args.split))
    if not split.has_electrons:
        raise SystemExit(f"{args.pack} has no electron arrays (made before them); pack it again")
    names = ("track_info", "track_mu_dist", "track_mu_pt", "track_e_angle", "track_e_d0", "track_e_pt")
    t = {n: np.asarray(split.array(n)) for n in names}
    ele_pt = np.asarray(split.array("ele_pt"))

    cand = (t["track_info"][:, 0] > args.pt_min) & ~(t["track_mu_dist"] < args.match_tol) \
        & (t["track_e_pt"] > args.pt_min)
    angle, d0 = t["track_e_angle"][cand], t["track_e_d0"][cand]
    label = packed.is_electron_track(t, args.pt_min, args.match_tol, args.e_angle, args.e_d0)
    n_ele = int((ele_pt > args.pt_min).sum())
    n_label = int(label.sum())
    result = {
        "pack": os.path.abspath(args.pack), "split": args.split, "events": len(split),
        "options": {k: getattr(args, k) for k in ("pt_min", "match_tol", "e_angle", "e_d0")},
        "candidate_tracks": int(cand.sum()),
        "angle": histogram(angle, ANGLE_EDGES),
        "d0_inside_angle": histogram(d0[angle < args.e_angle], D0_EDGES),
        "electrons": n_ele, "label_tracks": n_label,
        "label_tracks_below_pt_min": int((label & (t["track_info"][:, 0] <= args.pt_min)).sum()),
        "events_with_electron": int((ele_pt > args.pt_min).any(1).sum()),
    }

    def show(title, rows, unit):
        print(title)
        top = max([r["count"] for r in rows] + [1])
        for r in rows:
            print(f"  {r['low']:9.2g} - {r['high']:<9.2g}{unit} {r['count']:9d}  {'#' * round(40 * r['count'] / top)}")

    print(f"{args.pack} [{args.split}]: {len(split)} events, {result['events_with_electron']} with a GSF "
          f"electron above {args.pt_min:g} GeV ({n_ele} electrons)")
    print(f"candidate tracks (pT > {args.pt_min:g}, not a tracker muon, closest electron above "
          f"{args.pt_min:g}): {result['candidate_tracks']}")
    show("angle to the closest electron's inner direction:", result["angle"], " rad")
    show(f"inner point distance, tracks with angle < {args.e_angle:g}:", result["d0_inside_angle"], " m")
    print(f"electron label (angle < {args.e_angle:g}, d0 < {args.e_d0:g}): {n_label} tracks for {n_ele} "
          f"electrons ({n_label / max(n_ele, 1):.3f} per electron; "
          f"{result['label_tracks_below_pt_min']} of the tracks have pT <= {args.pt_min:g})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
