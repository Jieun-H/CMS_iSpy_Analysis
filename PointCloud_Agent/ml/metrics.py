"""ROC AUC, average precision and the per-split summary used by ml/train.py and ml/baseline.py
(numpy only, no sklearn)."""
import os

import numpy as np


def roc_auc(score, label):
    """Mann-Whitney AUC with tied scores averaged; None if one class is missing."""
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if not n_pos or not n_neg:
        return None
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s))
    _, start, count = np.unique(s, return_index=True, return_counts=True)
    for a, c in zip(start, count):
        ranks[a:a + c] = a + (c + 1) / 2
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(score, label):
    label = np.asarray(label).astype(bool)
    if not label.any():
        return None
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hit = label[order]
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float(precision[hit].mean())


def summarise(score, label, pt, pt_min, task):
    out = {"n": int(len(label)), "n_pos": int(np.sum(label)),
           "auc": roc_auc(score, label), "ap": average_precision(score, label),
           "acc@0.5": float(np.mean((score > 0.5) == (label > 0.5))) if len(label) else None}
    if task == "track":
        # tracks below pt_min can never be labelled; the hard part is among the rest
        hi = pt > pt_min
        out.update({f"{k}_pt": v for k, v in (("n", int(hi.sum())), ("n_pos", int(label[hi].sum())),
                                              ("auc", roc_auc(score[hi], label[hi])),
                                              ("ap", average_precision(score[hi], label[hi])))})
    return out


def run_pack(config, given=None):
    """Pack of a run: --pack if given, else the one recorded in its config.json. Runs made by
    condor before --pack-origin existed recorded the job's scratch copy, which is gone."""
    path = given or config["pack"]["path"]
    if not os.path.isfile(os.path.join(path, "pack.json")):
        raise SystemExit(f"pack {path} not found (a condor scratch copy?); give it with --pack (ml.report) / --ref-pack (ml.evaluate)")
    return path


def roc_curve(score, label):
    """-> efficiency of the positives (tpr), fraction of negatives passing (fpr), thresholds,
    one point per distinct score, from the highest threshold down."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(bool)
    order = np.argsort(-score, kind="mergesort")
    s, lab = score[order], label[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]          # last index of each distinct score
    tp = np.cumsum(lab)[last]
    fp = np.cumsum(~lab)[last]
    return tp / max(lab.sum(), 1), fp / max((~lab).sum(), 1), s[last]


def threshold_at_efficiency(score, label, efficiency):
    """Score cut keeping `efficiency` of the positives (score >= cut passes)."""
    pos = np.sort(np.asarray(score)[np.asarray(label).astype(bool)])
    if not len(pos):
        return None
    k = int(np.floor((1 - efficiency) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def pass_rate(score, mask, cut):
    """Fraction of the selected objects with score >= cut (None if none selected)."""
    mask = np.asarray(mask).astype(bool)
    n = int(mask.sum())
    return {"n": n, "rate": float((np.asarray(score)[mask] >= cut).mean()) if n else None}


def bootstrap(fn, groups, n=200, seed=0):
    """Std of fn(selected indices) over n resamplings of whole groups (e.g. events), so the
    tracks of one event stay together. groups: (N,) group id per object."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        v = fn(np.concatenate([members[g] for g in pick]))
        if v is not None:
            values.append(v)
    return float(np.std(values)) if values else None
