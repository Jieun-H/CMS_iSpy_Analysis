"""
Point-cloud npz (agent/pointcloud.py) -> model inputs, track by track.

track_id / muon_id are only list positions inside one ig collection. They are used here
to group the 33 Bezier points of each track and are then dropped, and the track order
in each event is shuffled, so neither the id nor a track's position in the array tells
the model anything. Same for ECAL hits (detid is dropped, hit order is shuffled).

Inputs   all tracks (tracks key: trackExtrapolator, pT > 1 GeV) + ECAL EB/EE hits.
         Reconstructed physics objects (elec_tracks, muon_tracks, global_muon,
         standalone_muon) and muon_chambers (only chambers matched to a reco muon)
         are never inputs: they are the answer.
Label    has_muon: at least one global muon above pt_min, taken from global_muon.

    ds = NpzEventDataset(glob.glob("out/*/pointcloud/Run_*/*.npz"), seed=0)
    batch = collate([ds[i] for i in range(8)])
    batch["track_points"]   # (B, T, 33, 3)  x,y,z along each track, inner -> outer
    batch["track_mask"]     # (B, T) True where a real track is
"""
import numpy as np

TRACK_POINT_FEATURES = ("x", "y", "z")                  # per point, in Bezier order
TRACK_FEATURES = ("pt",)                                # per track
HIT_FEATURES = ("x", "y", "z", "energy", "time")       # per ECAL hit (eta, phi follow from x, y, z)
HIT_KEYS = ("ecal_eb", "ecal_ee")                       # hit_det = position in this tuple
POINTS_PER_TRACK = 33

_DEFAULT_COLUMNS = {
    "tracks": ["x", "y", "z", "pt", "track_id"],
    "global_muon": ["x", "y", "z", "pt", "muon_id"],
    "ecal_eb": ["x", "y", "z", "energy", "eta", "phi", "time", "detid"],
    "ecal_ee": ["x", "y", "z", "energy", "eta", "phi", "time", "detid"],
}


def _columns(npz, key):
    name = f"{key}_columns"
    if name in npz.files:
        return [str(c) for c in npz[name]]
    return _DEFAULT_COLUMNS[key]


def _array(npz, key):
    """(rows, column index) for key; an absent key reads as empty."""
    cols = _columns(npz, key)
    arr = npz[key] if key in npz.files else np.zeros((0, len(cols)))
    return arr, {c: i for i, c in enumerate(cols)}


def _group_by_id(arr, id_col):
    """Rows of each object, in their original (along-the-track) order; object order is the id's."""
    ids = arr[:, id_col].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    uniq, starts = np.unique(ids[order], return_index=True)
    return np.split(arr[order], starts[1:]) if len(uniq) else []


def tracks(npz):
    """-> points (T, 33, 3) float32, features (T, len(TRACK_FEATURES)) float32"""
    arr, c = _array(npz, "tracks")
    groups = _group_by_id(arr, c["track_id"])
    bad = [len(g) for g in groups if len(g) != POINTS_PER_TRACK]
    if bad:
        raise ValueError(f"tracks with {sorted(set(bad))} points, expected {POINTS_PER_TRACK}")
    p_cols = [c[f] for f in TRACK_POINT_FEATURES]
    f_cols = [c[f] for f in TRACK_FEATURES]
    points = np.array([g[:, p_cols] for g in groups], dtype=np.float32).reshape(
        -1, POINTS_PER_TRACK, len(p_cols))
    feats = np.array([g[0, f_cols] for g in groups], dtype=np.float32).reshape(-1, len(f_cols))
    return points, feats


def hits(npz):
    """-> features (H, len(HIT_FEATURES)) float32, detector (H,) int64 (index into HIT_KEYS)"""
    feats, det = [], []
    for d, key in enumerate(HIT_KEYS):
        arr, c = _array(npz, key)
        feats.append(arr[:, [c[f] for f in HIT_FEATURES]].astype(np.float32))
        det.append(np.full(len(arr), d, dtype=np.int64))
    return np.concatenate(feats), np.concatenate(det)


def muons(npz, pt_min):
    """pT of the global muons above pt_min (label material, never an input)."""
    arr, c = _array(npz, "global_muon")
    pts = np.array([g[0, c["pt"]] for g in _group_by_id(arr, c["muon_id"])], dtype=np.float32)
    return pts[pts > pt_min]


def event(path, rng=None, pt_min=10.0):
    """One npz -> dict of inputs and label. rng: shuffle tracks and hits (None keeps file order)."""
    with np.load(path) as npz:
        track_points, track_feats = tracks(npz)
        hit_feats, hit_det = hits(npz)
        muon_pt = muons(npz, pt_min)
        ident = tuple(int(npz[k][0]) for k in ("run", "lumi", "event") if k in npz.files)
    if rng is not None:
        t = rng.permutation(len(track_points))
        track_points, track_feats = track_points[t], track_feats[t]
        h = rng.permutation(len(hit_feats))
        hit_feats, hit_det = hit_feats[h], hit_det[h]
    return {
        "track_points": track_points, "track_feats": track_feats,
        "hit_feats": hit_feats, "hit_det": hit_det,
        "has_muon": np.int64(len(muon_pt) > 0), "n_muons": np.int64(len(muon_pt)),
        "id": ident, "path": str(path),
    }


class NpzEventDataset:
    """Sequence of events; works as a torch Dataset (len + getitem) with collate_fn=collate.

    Tracks and hits are reshuffled on every access. With DataLoader workers, give each
    worker its own seed (worker_init_fn) so the workers do not repeat the same order.
    """

    def __init__(self, paths, seed=None, shuffle=True, pt_min=10.0):
        self.paths = sorted(str(p) for p in paths)
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.pt_min = pt_min

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return event(self.paths[i], self.rng if self.shuffle else None, self.pt_min)


def _pad(arrays, fill=0):
    """List of (n_i, ...) -> (B, max n_i, ...), mask (B, max n_i)."""
    n = max((len(a) for a in arrays), default=0)
    out = np.full((len(arrays), n) + arrays[0].shape[1:], fill, dtype=arrays[0].dtype)
    mask = np.zeros((len(arrays), n), dtype=bool)
    for b, a in enumerate(arrays):
        out[b, :len(a)] = a
        mask[b, :len(a)] = True
    return out, mask


def collate(events):
    """List of event() dicts -> padded numpy batch (torch.from_numpy each array if needed)."""
    track_points, track_mask = _pad([e["track_points"] for e in events])
    track_feats, _ = _pad([e["track_feats"] for e in events])
    hit_feats, hit_mask = _pad([e["hit_feats"] for e in events])
    hit_det, _ = _pad([e["hit_det"] for e in events])
    return {
        "track_points": track_points, "track_feats": track_feats, "track_mask": track_mask,
        "hit_feats": hit_feats, "hit_det": hit_det, "hit_mask": hit_mask,
        "has_muon": np.array([e["has_muon"] for e in events]),
        "n_muons": np.array([e["n_muons"] for e in events]),
    }
