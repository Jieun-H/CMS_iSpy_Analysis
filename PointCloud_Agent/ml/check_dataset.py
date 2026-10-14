"""
Check a point-cloud production before training (HANDOFF.md section 13, item 1).

Reads every npz through ml/inputs.py (the same path training uses), so a file that would
break training shows up here. Reports: npz count vs manifest, unreadable files, duplicate
(run, lumi, event), has_muon fraction, per-event size of each input, and how well the
track_is_muon matching works (end-point distance from each tracker muon above pt_min to
the closest track). numpy only, Python 3.9 (lxplus).

    python3 -m ml.check_dataset '/eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_UL16_train'
    python3 -m ml.check_dataset OUT --pt-min 20 --hit-keys ecal_eb ecal_ee --max-events 1000
    python3 -m ml.check_dataset OUT --json check.json      # also write the numbers to a file
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.inputs import HIT_KEYS, MATCH_TOL, event, match_tracks, tracks  # noqa: E402


def _npz_paths(target):
    """Production folder (OUT/pointcloud/Run_*/*.npz), a folder of npz, or a glob."""
    if os.path.isdir(target):
        for pattern in ("pointcloud/Run_*/*.npz", "Run_*/*.npz", "*.npz"):
            paths = glob.glob(os.path.join(target, pattern))
            if paths:
                return sorted(paths)
        return []
    return sorted(glob.glob(target))


def _manifest(target):
    path = os.path.join(target, "logs", "manifest.json") if os.path.isdir(target) else None
    if path and os.path.exists(path):
        with open(path) as f:
            return path, json.load(f)
    return None, None


def _summary(values):
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return {}
    return {"nonzero": int((v > 0).sum()), "mean": float(v.mean()), "min": float(v.min()),
            "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max())}


def _muon_match_distances(path, pt_min):
    """Each tracker muon above pt_min -> end-point distance (m) to the closest track."""
    with np.load(path) as npz:
        track_points, _ = tracks(npz)
        muon_points, muon_feats = tracks(npz, "muon_tracks")
    muon_points = muon_points[muon_feats[:, 0] > pt_min]
    return match_tracks(muon_points, track_points)[0].tolist()


def check(paths, pt_min, hit_keys, match_tol=MATCH_TOL):
    sizes = collections.defaultdict(list)
    hit_det = collections.Counter()
    labels, n_muons, muon_dist, n_muon_tracks = [], [], [], []
    ids = collections.defaultdict(list)
    errors = []
    for i, path in enumerate(paths):
        try:
            e = event(path, None, pt_min, hit_keys, match_tol)
            dist = _muon_match_distances(path, pt_min)
        except Exception as exc:  # report every bad file, keep going
            errors.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ids[e["id"]].append(path)
        sizes["tracks"].append(len(e["track_points"]))
        sizes["hits"].append(len(e["hit_feats"]))
        sizes["csc_segments"].append(int((e["seg_det"] == 0).sum()))
        sizes["dt_segments"].append(int((e["seg_det"] == 1).sum()))
        sizes["rpc_hits"].append(len(e["rpc_feats"]))
        hit_det.update(e["hit_det"].tolist())
        labels.append(int(e["has_muon"]))
        n_muons.append(int(e["n_muons"]))
        muon_dist.extend(dist)
        n_muon_tracks.append(int(e["track_is_muon"].sum()))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(paths)}", file=sys.stderr)
    n_ok = len(labels)
    return {
        "npz": len(paths),
        "read_ok": n_ok,
        "errors": errors,
        "duplicates": {str(k): v for k, v in ids.items() if len(v) > 1},
        "has_muon": {"pt_min": pt_min, "events": int(sum(labels)),
                     "fraction": float(np.mean(labels)) if n_ok else None,
                     "n_muons": dict(sorted(collections.Counter(n_muons).items()))},
        "per_event": {k: _summary(v) for k, v in sizes.items() if v},
        "track_is_muon": _match_report(muon_dist, n_muon_tracks, n_muons, match_tol),
        "hits_by_key": {key: {"total": int(hit_det[d]),
                              "per_event": hit_det[d] / n_ok if n_ok else None}
                        for d, key in enumerate(hit_keys)},
    }


def _match_report(dist, n_matched, n_global, tol):
    d = np.asarray(dist, dtype=np.float64)
    finite = d[np.isfinite(d)]
    edges = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    return {
        "match_tol": tol,
        "tracker_muons": len(d),
        "matched": int((d < tol).sum()),
        "no_track_in_event": int((~np.isfinite(d)).sum()),
        "distance_le": {f"{e:g}": int((finite <= e).sum()) for e in edges},
        "median_m": float(np.median(finite)) if len(finite) else None,
        # events where the per-track label and the event label disagree in count
        "events_matched_ne_global": int(sum(a != b for a, b in zip(n_matched, n_global))),
        "tracks_labelled": int(sum(n_matched)),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="production folder (has pointcloud/ and logs/), npz folder, or quoted glob")
    p.add_argument("--pt-min", type=float, default=10.0, help="global muon pT for has_muon (default 10)")
    p.add_argument("--hit-keys", nargs="+", default=list(HIT_KEYS), help="calorimeter keys to count")
    p.add_argument("--match-tol", type=float, default=MATCH_TOL,
                   help=f"track <-> tracker muon end-point distance in m (default {MATCH_TOL:g})")
    p.add_argument("--max-events", type=int, default=None, help="read only the first N npz (sorted)")
    p.add_argument("--json", help="also write the report to this file")
    args = p.parse_args(argv)

    paths = _npz_paths(args.target)
    if not paths:
        print(f"no npz found for {args.target}", file=sys.stderr)
        return 1
    if args.max_events:
        paths = paths[:args.max_events]
    report = check(paths, args.pt_min, tuple(args.hit_keys), args.match_tol)

    mpath, manifest = _manifest(args.target)
    if manifest is not None:
        report["manifest"] = {"path": mpath, "events": len(manifest.get("events", [])),
                              "requested": manifest.get("options", {}).get("nevents")}

    print(f"npz files      : {report['npz']}  (read OK {report['read_ok']}, errors {len(report['errors'])})")
    if "manifest" in report:
        m = report["manifest"]
        print(f"manifest       : {m['events']} events written, {m['requested']} requested")
    print(f"duplicate ids  : {len(report['duplicates'])}")
    h = report["has_muon"]
    if h["fraction"] is not None:
        print(f"has_muon       : {h['events']}/{report['read_ok']} = {h['fraction']:.3f}  "
              f"(global muon pT > {h['pt_min']:g})   n_muons {h['n_muons']}")
    print(f"{'per event':15s}  {'nonzero':>8s} {'mean':>8s} {'p50':>6s} {'p99':>7s} {'max':>6s}")
    for k, s in report["per_event"].items():
        print(f"{k:15s}  {s['nonzero']:8d} {s['mean']:8.1f} {s['p50']:6.0f} {s['p99']:7.0f} {s['max']:6.0f}")
    for k, s in report["hits_by_key"].items():
        if s["per_event"] is not None:
            print(f"  {k:13s}  {s['per_event']:8.1f} per event")
    t = report["track_is_muon"]
    print(f"track_is_muon  : {t['matched']}/{t['tracker_muons']} tracker muons (pT > {h['pt_min']:g}) "
          f"matched within {t['match_tol']:g} m, median distance {t['median_m']}")
    print(f"  distance <=  : {t['distance_le']}")
    print(f"  {t['tracks_labelled']} tracks labelled; {t['events_matched_ne_global']} events where "
          f"labelled tracks != global muons")
    for e in report["errors"][:10]:
        print(f"ERROR {e['path']}: {e['error']}")
    for k, v in list(report["duplicates"].items())[:10]:
        print(f"DUPLICATE {k}: {v}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if not report["errors"] and not report["duplicates"] else 2


if __name__ == "__main__":
    sys.exit(main())
