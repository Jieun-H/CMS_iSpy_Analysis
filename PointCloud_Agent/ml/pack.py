"""
Pack point-cloud npz into train / val / test splits for ml/train.py (numpy only, Python 3.9).

Reading 20k small npz from EOS every epoch is slow; this reads them once and writes a few
flat binary arrays per split (ml/packed.py). Everything an experiment may choose (track
features, calorimeter parts, label pt_min / match_tol) is kept, so pack once and choose at
training time. Normalisation sums are taken from the train split only.

    python3 -m ml.pack /eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_UL16_train \\
        --out /eos/user/j/jieun/ml/packed/zmm_ul16_v1
    python3 -m ml.pack OUT_A OUT_B --out PACK --split 0.7 0.15 0.15 --seed 0
    python3 -m ml.pack OUT --out PACK --max-events 500          # small pack for a quick test

Inputs may be production folders (OUT/pointcloud/Run_*/*.npz), npz folders or quoted globs.
The split is a seeded random permutation of the sorted paths; the path lists are saved in
each split's meta.json and pack.json.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import inputs, packed  # noqa: E402
from ml.check_dataset import _npz_paths  # noqa: E402

SPLITS = ("train", "val", "test")


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--", "."],
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def split_paths(paths, fractions, seed):
    order = np.random.default_rng(seed).permutation(len(paths))
    cuts = np.round(np.cumsum(fractions) / np.sum(fractions) * len(paths)).astype(int)
    parts = np.split(order, cuts[:-1])
    return {name: [paths[i] for i in sorted(part)] for name, part in zip(SPLITS, parts)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+", help="production folders, npz folders or quoted globs")
    p.add_argument("--out", required=True, help="pack folder (one sub-folder per split)")
    p.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"), help="fractions (default 0.7 0.15 0.15)")
    p.add_argument("--seed", type=int, default=0, help="split permutation seed (default 0)")
    p.add_argument("--max-events", type=int, default=None, help="use only the first N sorted npz")
    args = p.parse_args(argv)

    paths = sorted({q for t in args.targets for q in _npz_paths(t)})
    if not paths:
        print(f"no npz found in {args.targets}", file=sys.stderr)
        return 1
    if args.max_events:
        paths = paths[:args.max_events]
    splits = split_paths(paths, args.split, args.seed)
    print(" ".join(f"{k} {len(v)}" for k, v in splits.items()), file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    stats = packed.RunningStats()
    summary = {}
    for name in SPLITS:
        writer = packed.SplitWriter(os.path.join(args.out, name))
        n_tracks = n_labelled = 0
        for i, path in enumerate(splits[name]):
            raw = packed.read_raw(path)
            writer.add(path, raw)
            if name == "train":
                stats.add_event(raw)
            n_tracks += int(raw["n_tracks"])
            n_labelled += int(((raw["track_mu_dist"] < inputs.MATCH_TOL)
                               & (raw["track_mu_pt"] > 10)).sum())
            if (i + 1) % 2000 == 0:
                print(f"  {name} {i + 1}/{len(splits[name])}", file=sys.stderr)
        writer.close({"split": name})
        summary[name] = {"events": len(splits[name]), "tracks": n_tracks,
                         "muon_tracks_pt10": n_labelled}
        print(f"{name}: {summary[name]}", file=sys.stderr)

    info = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "git": git_info(),
        "command": sys.argv,
        "targets": args.targets, "split": list(args.split), "seed": args.seed,
        "track_info_features": list(inputs.TRACK_INFO_FEATURES),
        "track_scalars": packed.ALL_TRACK_SCALARS,
        "hit_keys": list(inputs.HIT_KEYS), "seg_keys": list(inputs.SEG_KEYS),
        "summary": summary,
        "stats": stats.to_json(),
        "paths": splits,
    }
    with open(os.path.join(args.out, "pack.json"), "w") as f:
        json.dump(info, f)
    print(f"done -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
