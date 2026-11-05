"""
Per GSF electron: is its track among the model's input `tracks`, and which track is it?
Reads the npz directly (the pack keeps only per-track matches, not the electrons).

The electron label (ml/packed.py is_electron_track) assumes that every GSF electron has an
ordinary track (generalTracks, Kalman fit) in `tracks` that starts at the same point in the
same direction, and that the GSF track itself (refitted, Gaussian Sum Filter) is not in
`tracks`. For every GSF electron with pT > pt_min it prints
    same track     both end points within --same-tol of a track: the GSF track itself is
                   among the inputs (expected: none; else the label leaks into the inputs)
    matched        tracks inside --e-angle / --e-d0 (0, 1, 2+): 1 is the ideal; 0 means no
                   input track belongs to the electron (e.g. ECAL-seeded, track lost)
    nearest        start-direction angle, start-point distance, end-point distance and
                   pT ratio (track / electron) of the closest track by angle
and the same for tracker muons as a reference (their track is taken as is: same track).

numpy only (Python 3.9), no GPU:
    python3 -m ml.electron_tracks /eos/user/j/jieun/PointCloud_Agent_output/DYJetsToLL_UL16_train --max-files 2000
"""
import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import inputs, packed  # noqa: E402
from ml.check_dataset import _npz_paths  # noqa: E402

QUANTILES = (0.1, 0.5, 0.9, 0.99)


def _direction(points):
    d = points[:, 1] - points[:, 0]
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def per_object(t_points, t_pt, o_points, o_pt):
    """For each object (GSF electron / tracker muon) against the event's tracks: end-point
    distance to the closest track, and angle, start distance, end distance and pT ratio of
    the closest track by angle, plus the (O, T) angle and start-distance matrices."""
    cos = _direction(o_points.astype(np.float64)) @ _direction(t_points.astype(np.float64)).T
    angle = np.arccos(np.clip(cos, -1, 1))                                       # (O, T)
    start = np.linalg.norm(o_points[:, None, 0] - t_points[None, :, 0], axis=-1)   # (O, T)
    end = np.linalg.norm(o_points[:, None, -1] - t_points[None, :, -1], axis=-1)
    same = np.maximum(start, end).min(1)
    j = angle.argmin(1)
    i = np.arange(len(o_points))
    return same, angle[i, j], start[i, j], end[i, j], t_pt[j] / np.maximum(o_pt, 1e-9), angle, start


def quantiles(x):
    x = np.asarray(x)
    return {str(q): float(np.quantile(x, q)) for q in QUANTILES} if len(x) else {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="production folder, npz folder or quoted glob (as ml/check_dataset.py)")
    p.add_argument("--max-files", type=int, default=2000, help="first N npz (default 2000, 0 = all)")
    p.add_argument("--pt-min", type=float, default=10.0)
    p.add_argument("--e-angle", type=float, default=packed.E_ANGLE)
    p.add_argument("--e-d0", type=float, default=packed.E_D0)
    p.add_argument("--same-tol", type=float, default=inputs.MATCH_TOL,
                   help="both end points within this (m) = the same track (default as for muons)")
    p.add_argument("--json", help="also write the result to this file")
    args = p.parse_args(argv)

    paths = _npz_paths(args.target)
    if args.max_files:
        paths = paths[:args.max_files]
    if not paths:
        raise SystemExit(f"no npz found in {args.target}")
    pt_i = inputs.TRACK_FEATURES.index("pt")
    acc = {k: {"same": [], "angle": [], "start": [], "end": [], "pt_ratio": [], "n_match": []}
           for k in ("electron", "muon")}
    n_no_tracks = 0
    for n, path in enumerate(paths, 1):
        with np.load(path) as npz:
            t_points, t_feats = inputs.tracks(npz, "tracks")
            objs = {"electron": inputs.tracks(npz, "elec_tracks"), "muon": inputs.tracks(npz, "muon_tracks")}
        for kind, (o_points, o_feats) in objs.items():
            keep = o_feats[:, pt_i] > args.pt_min if len(o_feats) else np.zeros(0, bool)
            if not keep.any():
                continue
            if not len(t_points):
                n_no_tracks += int(keep.sum())
                continue
            same, angle, start, end, ratio, a_all, s_all = per_object(
                t_points, t_feats[:, pt_i], o_points[keep], o_feats[keep, pt_i])
            a = acc[kind]
            a["same"] += same.tolist()
            a["angle"] += angle.tolist()
            a["start"] += start.tolist()
            a["end"] += end.tolist()
            a["pt_ratio"] += ratio.tolist()
            a["n_match"] += ((a_all < args.e_angle) & (s_all < args.e_d0)).sum(1).tolist()
        if n % 500 == 0:
            print(f"  {n}/{len(paths)}", file=sys.stderr)

    result = {"target": args.target, "files": len(paths),
              "options": {k: getattr(args, k) for k in ("pt_min", "e_angle", "e_d0", "same_tol")},
              "objects_in_events_without_tracks": n_no_tracks}
    for kind, a in acc.items():
        same, n_match = np.array(a["same"]), np.array(a["n_match"])
        result[kind] = {
            "objects": len(same),
            "same_track": int((same < args.same_tol).sum()),
            "matched": {"0": int((n_match == 0).sum()), "1": int((n_match == 1).sum()),
                        "2+": int((n_match >= 2).sum())},
            "nearest_angle": quantiles(a["angle"]), "nearest_start": quantiles(a["start"]),
            "nearest_end": quantiles(a["end"]), "nearest_pt_ratio": quantiles(a["pt_ratio"]),
            "same_distance": quantiles(same),
        }

    print(f"{args.target}: {len(paths)} npz, pT > {args.pt_min:g}, match = angle < {args.e_angle:g} rad "
          f"and start < {args.e_d0:g} m, same track = both ends < {args.same_tol:g} m")
    for kind in ("electron", "muon"):
        r = result[kind]
        n = max(r["objects"], 1)
        name = "GSF electrons" if kind == "electron" else "tracker muons (reference)"
        print(f"{name}: {r['objects']}")
        print(f"  same track in `tracks`  : {r['same_track']} ({r['same_track'] / n:.3f})")
        print("  matched tracks 0 / 1 / 2+: " + " / ".join(
            f"{v} ({v / n:.3f})" for v in r["matched"].values()))
        for q in ("nearest_angle", "nearest_start", "nearest_end", "nearest_pt_ratio", "same_distance"):
            print(f"  {q:17s} " + "  ".join(f"q{k}={v:.3g}" for k, v in r[q].items()))
    if n_no_tracks:
        print(f"objects in events without tracks: {n_no_tracks}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
