"""
Single-variable baselines for the per-track muon label, on a pack made by ml/pack.py.

A trained model is only interesting if it beats what one simple variable already gives.
In Z -> mumu most high-pT tracks are the muons, so pT alone separates well; isolation and
"muon-system hits around the track" are the classic hand-made muon variables. Each
variable is used directly as the score (both signs tried, the better one reported) on the
tracks with pT > pt_min, the same tracks as auc_pt / ap_pt in ml/train.py.

Variables (cones in eta-phi, phi wrapped; all cone sizes are options):
    pt, abs_eta, chi2_ndof, ndof              track_info
    iso_rel     sum pT of other tracks within --iso-cone of the track direction / pT
    n_seg       CSC + DT segments within --muon-cone of the track direction
    n_rpc       RPC hits within --muon-cone of the track direction
    ecal_e      ECAL (EB + EE) energy within --ecal-cone of the track's outer end point

numpy only (Python 3.9), no GPU:
    python3 -m ml.baseline --pack /eos/user/j/jieun/ml/packed/zmm_ul16_v1
    python3 -m ml.baseline --pack PACK --split val --pt-min 20 --muon-cone 0.2 --json base.json
"""
import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import inputs, packed  # noqa: E402
from ml.metrics import average_precision, roc_auc  # noqa: E402

VARIABLES = ("pt", "abs_eta", "chi2_ndof", "ndof", "iso_rel", "n_seg", "n_rpc", "ecal_e")


def _dr2(a, b):
    """(n, 2) and (m, 2) eta-phi -> (n, m) squared distance, phi wrapped."""
    deta = a[:, None, 0] - b[None, :, 0]
    dphi = np.remainder(a[:, None, 1] - b[None, :, 1] + np.pi, 2 * np.pi) - np.pi
    return deta ** 2 + dphi ** 2


def event_variables(raw, args):
    """Raw packed event -> (label, variables (T, len(VARIABLES))) for the tracks above pt_min."""
    info = raw["track_info"]
    c = {f: info[:, i] for i, f in enumerate(inputs.TRACK_INFO_FEATURES)}
    keep = c["pt"] > args.pt_min
    label = ((raw["track_mu_dist"] < args.match_tol) & (raw["track_mu_pt"] > args.pt_min))[keep]
    if not keep.any():
        return label, np.zeros((0, len(VARIABLES)))
    direction = np.stack([c["eta"], c["phi"]], -1)
    d_sel = direction[keep]
    pt = c["pt"]
    # isolation: other tracks (any pT) around the track direction
    near = _dr2(d_sel, direction) < args.iso_cone ** 2
    iso = (near * pt[None]).sum(1) - pt[keep]
    seg = packed.seg_transform(raw["seg_feats"])
    n_seg = (_dr2(d_sel, packed.eta_phi(seg[:, :3])) < args.muon_cone ** 2).sum(1) if len(seg) else 0
    rpc = raw["rpc_feats"]
    n_rpc = (_dr2(d_sel, packed.eta_phi(rpc)) < args.muon_cone ** 2).sum(1) if len(rpc) else 0
    h, det = raw["hit_feats"], raw["hit_det"]
    ecal = np.isin(det, [inputs.HIT_KEYS.index("ecal_eb"), inputs.HIT_KEYS.index("ecal_ee")])
    h = h[ecal]
    outer = packed.eta_phi(raw["track_points"][keep, -1])
    ecal_e = ((_dr2(outer, packed.eta_phi(h[:, :3])) < args.ecal_cone ** 2) * h[None, :, 3]).sum(1) \
        if len(h) else 0
    cols = {"pt": pt[keep], "abs_eta": np.abs(c["eta"][keep]),
            "chi2_ndof": c["chi2"][keep] / np.maximum(c["ndof"][keep], 1), "ndof": c["ndof"][keep],
            "iso_rel": iso / pt[keep], "n_seg": n_seg, "n_rpc": n_rpc, "ecal_e": ecal_e}
    return label, np.stack([np.broadcast_to(np.asarray(cols[v], dtype=np.float64), label.shape)
                            for v in VARIABLES], -1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, help="folder made by ml/pack.py")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--pt-min", type=float, default=10.0, help="label pT and track selection (default 10)")
    p.add_argument("--match-tol", type=float, default=inputs.MATCH_TOL, help="label match distance in m")
    p.add_argument("--iso-cone", type=float, default=0.3, help="isolation cone (default 0.3)")
    p.add_argument("--muon-cone", type=float, default=0.3, help="segment / RPC cone (default 0.3)")
    p.add_argument("--ecal-cone", type=float, default=0.1, help="ECAL cone at the outer end point (default 0.1)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--json", help="also write the result to this file")
    args = p.parse_args(argv)

    split = packed.PackedSplit(os.path.join(args.pack, args.split))
    n = len(split) if args.max_events is None else min(args.max_events, len(split))
    labels, values = [], []
    for e in range(n):
        lab, val = event_variables(split.event(e), args)
        labels.append(lab)
        values.append(val)
    label = np.concatenate(labels)
    value = np.concatenate(values)

    result = {"split": args.split, "events": n, "tracks": int(len(label)), "muons": int(label.sum()),
              "muon_fraction": float(label.mean()) if len(label) else None,
              "options": {k: getattr(args, k) for k in ("pt_min", "match_tol", "iso_cone", "muon_cone",
                                                         "ecal_cone")},
              "variables": {}}
    print(f"{args.split}: {n} events, {len(label)} tracks with pT > {args.pt_min:g}, "
          f"{int(label.sum())} muons ({result['muon_fraction']:.3f} = AP of a random score)")
    print(f"{'variable':10s} {'sign':>4s} {'auc':>6s} {'ap':>6s}   muon mean / other mean")
    for i, v in enumerate(VARIABLES):
        x = value[:, i]
        auc = roc_auc(x, label)
        sign = 1 if auc is None or auc >= 0.5 else -1
        auc = roc_auc(sign * x, label)
        ap = average_precision(sign * x, label)
        result["variables"][v] = {"sign": sign, "auc": auc, "ap": ap,
                                  "mean_muon": float(x[label].mean()) if label.any() else None,
                                  "mean_other": float(x[~label].mean()) if (~label).any() else None}
        r = result["variables"][v]
        fmt = (lambda y: "  none" if y is None else f"{y:6.3f}")
        fmt_mean = (lambda y: "none" if y is None else f"{y:.3g}")
        print(f"{v:10s} {'+' if sign > 0 else '-':>4s} {fmt(auc)} {fmt(ap)}   "
              f"{fmt_mean(r['mean_muon'])} / {fmt_mean(r['mean_other'])}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
