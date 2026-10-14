"""
Packed training data: many npz -> a few flat binary arrays per split, plus the feature
transforms, normalisation and padding used to turn events into model inputs. numpy only
(Python 3.9), shared by ml/pack.py (writes) and ml/train.py (reads).

A packed split keeps everything an experiment may choose at training time, so one pack
serves all experiments:
  - every track feature in inputs.TRACK_INFO_FEATURES,
  - every calorimeter part in inputs.HIT_KEYS (choose with hit_keys at training),
  - per track: distance to the closest tracker muon and that muon's pT (label with any
    pt_min / match_tol at training), per event: global muon pTs (event label),
  - evaluation only, never a model input: per track the angle (rad) between the track's
    inner direction and the closest GSF electron's, the inner-point distance (m) and pT of
    that electron; per event the GSF electron pTs (ee / mumu / emu event categories).
    GSF tracks are refitted, so their end points do not coincide with the track's
    (unlike tracker muons); the direction at the inner point does. Packs made before this
    have no electron arrays; they read as "no electron information" (inf / 0).

Layout of <pack>/<split>/:
    meta.json            array dtypes and shapes, npz paths (event order), config
    <name>.bin           raw little-endian arrays (np.memmap)
Event e owns rows offsets[e]:offsets[e+1] of each per-object array (n_tracks etc.).
"""
import json
import os

import numpy as np

from ml import inputs

GLB_SLOTS = 4                                   # global muon pTs kept per event, highest first
PER_EVENT = ("n_tracks", "n_hits", "n_segs", "n_rpc", "glb_pt", "ele_pt")
PER_OBJECT = {                                  # array -> the per-event count that indexes it
    "track_points": "n_tracks", "track_info": "n_tracks",
    "track_mu_dist": "n_tracks", "track_mu_pt": "n_tracks",
    "track_e_angle": "n_tracks", "track_e_d0": "n_tracks", "track_e_pt": "n_tracks",
    "hit_feats": "n_hits", "hit_det": "n_hits",
    "seg_feats": "n_segs", "seg_det": "n_segs",
    "rpc_feats": "n_rpc",
}


# arrays added after the first packs; an old pack reads them with these fills
OPTIONAL = {"ele_pt": 0.0, "track_e_angle": np.inf, "track_e_d0": np.inf, "track_e_pt": 0.0}


# ---------------------------------------------------------------- reading npz (pack time)

def _inner_direction(points):
    d = points[:, 1] - points[:, 0]
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def match_electrons(points, e_points, e_pt):
    """Per track: angle (rad) to the closest GSF electron by inner direction, that electron's
    inner-point distance (m) and pT. inf / inf / 0 when the event has no electron."""
    n = len(points)
    if not n or not len(e_points):
        return np.full(n, np.inf, np.float32), np.full(n, np.inf, np.float32), np.zeros(n, np.float32)
    cos = _inner_direction(points) @ _inner_direction(e_points).T          # (T, E)
    angle = np.arccos(np.clip(cos, -1, 1))
    j = angle.argmin(axis=1)
    d0 = np.linalg.norm(points[:, 0] - e_points[j, 0], axis=1)
    return (angle[np.arange(n), j].astype(np.float32), d0.astype(np.float32),
            e_pt[j].astype(np.float32))


def read_raw(path):
    """One npz -> dict of raw arrays (no selection, no transform)."""
    with np.load(path) as npz:
        points, info = inputs.tracks(npz, "tracks", inputs.TRACK_INFO_FEATURES)
        mu_points, mu_feats = inputs.tracks(npz, "muon_tracks")
        dist, j = inputs.match_tracks(points, mu_points)
        mu_pt = np.where(j >= 0, mu_feats[np.maximum(j, 0), 0] if len(mu_feats) else 0, 0)
        hit_feats, hit_det = inputs.hits(npz, inputs.HIT_KEYS)
        seg_feats, seg_det = inputs.segments(npz)
        rpc = inputs.rpc_hits(npz)
        glb = np.sort(inputs.muons(npz, -np.inf))[::-1][:GLB_SLOTS]
        e_points, e_feats = inputs.tracks(npz, "elec_tracks")
        e_angle, e_d0, e_pt = match_electrons(points, e_points, e_feats[:, 0])
        ele = np.sort(e_feats[:, 0])[::-1][:GLB_SLOTS]
    glb_pt = np.zeros(GLB_SLOTS, dtype=np.float32)
    glb_pt[:len(glb)] = glb
    ele_pt = np.zeros(GLB_SLOTS, dtype=np.float32)
    ele_pt[:len(ele)] = ele
    return {
        "track_points": points, "track_info": info,
        "track_mu_dist": dist.astype(np.float32), "track_mu_pt": mu_pt.astype(np.float32),
        "hit_feats": hit_feats, "hit_det": hit_det.astype(np.int8),
        "seg_feats": seg_feats, "seg_det": seg_det.astype(np.int8),
        "rpc_feats": rpc,
        "n_tracks": np.int32(len(points)), "n_hits": np.int32(len(hit_feats)),
        "n_segs": np.int32(len(seg_feats)), "n_rpc": np.int32(len(rpc)), "glb_pt": glb_pt,
        "track_e_angle": e_angle, "track_e_d0": e_d0, "track_e_pt": e_pt, "ele_pt": ele_pt,
    }


# ---------------------------------------------------------------- feature transforms

def track_scalar_names(features):
    """Transformed per-track scalar columns for the chosen track_features."""
    names = []
    for f in features:
        names += {"pt": ["log_pt"], "phi": ["sin_phi", "cos_phi"], "chi2": ["log1p_chi2"],
                  "ndof": ["log1p_ndof"]}.get(f, [f])
    return names


ALL_TRACK_SCALARS = track_scalar_names(inputs.TRACK_INFO_FEATURES)


def track_scalars(info):
    """track_info (T, 6) in TRACK_INFO_FEATURES order -> (T, len(ALL_TRACK_SCALARS))."""
    c = {f: info[:, i] for i, f in enumerate(inputs.TRACK_INFO_FEATURES)}
    cols = {"log_pt": np.log(np.maximum(c["pt"], 1e-3)), "eta": c["eta"],
            "sin_phi": np.sin(c["phi"]), "cos_phi": np.cos(c["phi"]), "charge": c["charge"],
            "log1p_chi2": np.log1p(np.maximum(c["chi2"], 0)),
            "log1p_ndof": np.log1p(np.maximum(c["ndof"], 0))}
    return np.stack([cols[n] for n in ALL_TRACK_SCALARS], axis=-1).astype(np.float32)


HIT_NAMES = ("x", "y", "z", "log_energy", "time")
SEG_NAMES = ("mid_x", "mid_y", "mid_z", "dir_x", "dir_y", "dir_z")
OTHER_WIDTH = max(len(HIT_NAMES), len(SEG_NAMES), 3)    # non-track node feature width (zero-padded)


def hit_transform(h):
    out = h.copy()
    out[:, 3] = np.log(np.maximum(h[:, 3], 1e-3))
    return out


def seg_transform(s):
    """End points -> mid point, unit direction pointing away from the beam line's origin."""
    a, b = s[:, :3], s[:, 3:6]
    mid = (a + b) / 2
    d = b - a
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    d = d * np.where((d * mid).sum(axis=1, keepdims=True) < 0, -1, 1)
    return np.concatenate([mid, d], axis=1).astype(np.float32)


def eta_phi(xyz):
    """(N, 3) positions -> (N, 2) eta, phi (kNN coordinates of the first EdgeConv)."""
    r = np.maximum(np.hypot(xyz[:, 0], xyz[:, 1]), 1e-6)
    return np.stack([np.arcsinh(xyz[:, 2] / r), np.arctan2(xyz[:, 1], xyz[:, 0])], -1).astype(np.float32)


# ---------------------------------------------------------------- normalisation

class RunningStats:
    """Per-column sum / sum of squares / count, combinable across groups."""

    def __init__(self):
        self.groups = {}

    def add(self, group, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        s = self.groups.setdefault(group, [np.zeros(x.shape[1]), np.zeros(x.shape[1]), 0])
        s[0] += x.sum(0)
        s[1] += (x * x).sum(0)
        s[2] += len(x)

    def add_event(self, raw):
        self.add("track_points", raw["track_points"])
        self.add("track_scalars", track_scalars(raw["track_info"]))
        h = hit_transform(raw["hit_feats"])
        for d, key in enumerate(inputs.HIT_KEYS):
            self.add(f"hits/{key}", h[raw["hit_det"] == d])
        g = seg_transform(raw["seg_feats"])
        for d, key in enumerate(inputs.SEG_KEYS):
            self.add(f"segs/{key}", g[raw["seg_det"] == d])
        self.add("rpc", raw["rpc_feats"])

    def to_json(self):
        return {k: {"sum": v[0].tolist(), "sumsq": v[1].tolist(), "n": int(v[2])}
                for k, v in self.groups.items()}


def mean_std(stats, groups):
    """Combine groups of a stats json -> mean, std (float32). std floor 1e-6."""
    total = [0.0, 0.0, 0]
    for g in groups:
        if g in stats:
            total[0] = total[0] + np.array(stats[g]["sum"])
            total[1] = total[1] + np.array(stats[g]["sumsq"])
            total[2] += stats[g]["n"]
    if not total[2]:
        return None, None
    mean = total[0] / total[2]
    std = np.sqrt(np.maximum(total[1] / total[2] - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


# ---------------------------------------------------------------- packed split on disk

class SplitWriter:
    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.files, self.shapes, self.dtypes = {}, {}, {}
        self.paths = []

    def add(self, path, raw):
        self.paths.append(str(path))
        for name in PER_EVENT + tuple(PER_OBJECT):
            a = np.ascontiguousarray(raw[name])
            if name not in self.files:
                self.files[name] = open(os.path.join(self.folder, name + ".bin"), "wb")
                self.shapes[name] = [0] + list(a.shape[1:] if name in PER_OBJECT else a.shape)
                self.dtypes[name] = a.dtype.str
            if a.dtype.str != self.dtypes[name]:
                a = a.astype(self.dtypes[name])
            self.files[name].write(a.tobytes())
            self.shapes[name][0] += len(a) if name in PER_OBJECT else 1

    def close(self, extra):
        for f in self.files.values():
            f.close()
        meta = {"arrays": {n: {"dtype": self.dtypes[n], "shape": self.shapes[n]} for n in self.files},
                "paths": self.paths, **extra}
        with open(os.path.join(self.folder, "meta.json"), "w") as f:
            json.dump(meta, f)


class PackedSplit:
    """Read-only view of one packed split. Arrays are memory-mapped on first use, so it
    can be handed to DataLoader workers (each worker maps the files itself)."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "meta.json")) as f:
            self.meta = json.load(f)
        self.paths = self.meta["paths"]
        self._arrays = None
        self._offsets = None

    def __len__(self):
        return len(self.paths)

    def __getstate__(self):          # do not pickle the memmaps into workers
        return {"folder": self.folder, "meta": self.meta, "paths": self.paths,
                "_arrays": None, "_offsets": None}

    def array(self, name):
        if self._arrays is None:
            self._arrays = {}
            for n, a in self.meta["arrays"].items():
                shape = tuple(a["shape"])
                path = os.path.join(self.folder, n + ".bin")
                self._arrays[n] = (np.memmap(path, dtype=np.dtype(a["dtype"]), mode="r", shape=shape)
                                   if shape[0] else np.zeros(shape, dtype=np.dtype(a["dtype"])))
            self._offsets = {c: np.concatenate([[0], np.cumsum(self._arrays[c], dtype=np.int64)])
                             for c in ("n_tracks", "n_hits", "n_segs", "n_rpc")}
        return self._arrays[name]

    @property
    def has_electrons(self):
        """False for packs made before the electron arrays were added."""
        return "ele_pt" in self.meta["arrays"]

    def event(self, e):
        """Raw arrays of event e (same keys as read_raw)."""
        self.array("n_tracks")
        out = {}
        for n in PER_EVENT:
            out[n] = (np.asarray(self.array(n)[e]) if n in self._arrays
                      else np.full(GLB_SLOTS, OPTIONAL[n], np.float32))
        for n, count in PER_OBJECT.items():
            o = self._offsets[count]
            out[n] = (np.asarray(self.array(n)[o[e]:o[e + 1]]) if n in self._arrays
                      else np.full(o[e + 1] - o[e], OPTIONAL[n], np.float32))
        return out


# ---------------------------------------------------------------- model inputs

class InputConfig:
    """What the model sees and how it is labelled; chosen at training time, saved in checkpoints."""

    def __init__(self, track_features=inputs.TRACK_INFO_FEATURES,
                 hit_keys=("ecal_eb", "ecal_ee", "hcal_hb", "hcal_he"),
                 segments=True, rpc=True, pt_min=10.0, match_tol=inputs.MATCH_TOL, max_hits=None):
        unknown = set(track_features) - set(inputs.TRACK_INFO_FEATURES)
        unknown |= set(hit_keys) - set(inputs.HIT_KEYS)
        if unknown:
            raise ValueError(f"unknown features / hit keys: {sorted(unknown)}")
        self.track_features = tuple(track_features)
        self.hit_keys = tuple(hit_keys)
        self.segments = bool(segments)
        self.rpc = bool(rpc)
        self.pt_min = float(pt_min)
        self.match_tol = float(match_tol)
        self.max_hits = max_hits

    def to_json(self):
        return dict(vars(self))

    @property
    def node_types(self):
        """Names of the node types; type id = position."""
        return (["track"] + list(self.hit_keys) + (list(inputs.SEG_KEYS) if self.segments else [])
                + (["rpc_hits"] if self.rpc else []))

    @property
    def scalar_cols(self):
        names = track_scalar_names(self.track_features)
        return [ALL_TRACK_SCALARS.index(n) for n in names]


class Normalizer:
    """Mean / std per input group from the train split's sums (ml/pack.py). A trained model
    keeps its own: Normalizer.from_json(checkpoint["config"]["normalisation"], cfg)."""

    def __init__(self, stats, cfg):
        self.cfg = cfg
        self.track_points = mean_std(stats, ["track_points"])
        m, s = mean_std(stats, ["track_scalars"])
        cols = cfg.scalar_cols
        self.track_scalars = (m[cols], s[cols])
        self.hits = mean_std(stats, [f"hits/{k}" for k in cfg.hit_keys])
        self.segs = mean_std(stats, [f"segs/{k}" for k in inputs.SEG_KEYS])
        self.rpc = mean_std(stats, ["rpc"])

    @staticmethod
    def apply(x, ms):
        mean, std = ms
        if mean is None or not x.size:
            return x.astype(np.float32)
        return ((x - mean) / std).astype(np.float32)

    @classmethod
    def from_json(cls, saved, cfg):
        norm = cls.__new__(cls)
        norm.cfg = cfg
        for k, v in saved.items():
            setattr(norm, k, (None, None) if v is None else
                    (np.array(v["mean"], np.float32), np.array(v["std"], np.float32)))
        return norm

    def to_json(self):
        def j(ms):
            return None if ms[0] is None else {"mean": ms[0].tolist(), "std": ms[1].tolist()}
        return {k: j(getattr(self, k)) for k in ("track_points", "track_scalars", "hits", "segs", "rpc")}


def event_inputs(raw, cfg, norm):
    """Raw event -> per-type node arrays for the model (unpadded)."""
    c = cfg
    # tracks
    pts = raw["track_points"]
    scal = track_scalars(raw["track_info"])[:, c.scalar_cols]
    label = (raw["track_mu_dist"] < c.match_tol) & (raw["track_mu_pt"] > c.pt_min)
    out = {
        "track_points": norm.apply(pts, norm.track_points),
        "track_scalars": norm.apply(scal, norm.track_scalars),
        "track_coord": eta_phi(pts[:, -1]) if len(pts) else np.zeros((0, 2), np.float32),
        "track_label": label.astype(np.float32),
        "track_pt": raw["track_info"][:, 0].astype(np.float32),
        "track_eta": raw["track_info"][:, 1].astype(np.float32),
    }
    # calorimeter hits of the chosen parts; hit type = 1 + position in hit_keys
    h, det = raw["hit_feats"], raw["hit_det"].astype(np.int64)
    keep_det = [inputs.HIT_KEYS.index(k) for k in c.hit_keys]
    sel = np.isin(det, keep_det)
    h, det = h[sel], det[sel]
    if c.max_hits is not None and len(h) > c.max_hits:
        top = np.argsort(-h[:, 3])[:c.max_hits]
        h, det = h[top], det[top]
    remap = np.full(len(inputs.HIT_KEYS), -1)
    remap[keep_det] = np.arange(len(keep_det))
    nodes = [(norm.apply(hit_transform(h), norm.hits), 1 + remap[det], eta_phi(h[:, :3]))]
    n_type = 1 + len(c.hit_keys)
    if c.segments:
        g = raw["seg_feats"]
        gt = seg_transform(g)
        nodes.append((norm.apply(gt, norm.segs), n_type + raw["seg_det"].astype(np.int64),
                      eta_phi(gt[:, :3])))
        n_type += len(inputs.SEG_KEYS)
    if c.rpc:
        r = raw["rpc_feats"]
        nodes.append((norm.apply(r, norm.rpc), np.full(len(r), n_type, np.int64), eta_phi(r)))
    # non-track nodes of different widths are zero-padded to one width (type says which)
    feats = [np.pad(x, ((0, 0), (0, OTHER_WIDTH - x.shape[1]))) for x, _, _ in nodes]
    out["other_feats"] = np.concatenate(feats).astype(np.float32)
    out["other_type"] = np.concatenate([t for _, t, _ in nodes]).astype(np.int64)
    out["other_coord"] = np.concatenate([p for _, _, p in nodes]).astype(np.float32)
    out["has_muon"] = np.float32((raw["glb_pt"] > c.pt_min).any())
    return out


def pad_batch(events):
    """List of event_inputs dicts -> padded batch (numpy) with masks."""
    out = {}
    for key, mask_name in (("track_points", "track_mask"), ("track_scalars", None),
                           ("track_coord", None), ("track_label", None), ("track_pt", None), ("track_eta", None),
                           ("other_feats", "other_mask"), ("other_type", None), ("other_coord", None)):
        arrays = [e[key] for e in events]
        n = max(1, max(len(a) for a in arrays))
        pad = np.zeros((len(arrays), n) + arrays[0].shape[1:], dtype=arrays[0].dtype)
        mask = np.zeros((len(arrays), n), dtype=bool)
        for b, a in enumerate(arrays):
            pad[b, :len(a)] = a
            mask[b, :len(a)] = True
        out[key] = pad
        if mask_name:
            out[mask_name] = mask
    out["has_muon"] = np.array([e["has_muon"] for e in events], dtype=np.float32)
    return out
