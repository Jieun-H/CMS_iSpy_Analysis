"""
Point-cloud npz (agent/pointcloud.py) -> model inputs, track by track.

track_id / muon_id are only list positions inside one ig collection. They are used here
to group the 33 Bezier points of each track and are then dropped, and the track order
in each event is shuffled, so neither the id nor a track's position in the array tells
the model anything. Same for calorimeter hits, muon segments and RPC hits (detid and the
chamber numbers are dropped, the order is shuffled).

Inputs   all tracks (tracks key: trackExtrapolator, pT > 1 GeV) + calorimeter hits
         (ECAL EB/EE, HCAL HB/HE/HF/HO; choose with hit_keys=...) + CSC/DT segments + RPC hits.
         AOD keeps only the HCAL hits around reconstructed objects (reducedHcalRecHits),
         so HB/HE/HO are mostly empty; HF is filled in every event. Segments and RPC hits are local measurements in
         the muon system, made before (and independent of) muon reconstruction.
         Reconstructed physics objects (elec_tracks, muon_tracks, global_muon,
         standalone_muon) and muon_chambers (only chambers matched to a reco muon)
         are never inputs: they are the answer.
Label    has_muon: at least one global muon above pt_min, taken from global_muon.
         track_is_muon (per track): the track is the inner track of a tracker muon above
         pt_min. tracks and muon_tracks carry no common id, so a track is matched when both
         Bezier end points (the Extras_V1 pos_1, pos_2 the curve passes through) lie within
         match_tol (m) of a muon track's end points.

    ds = NpzEventDataset(glob.glob("out/*/pointcloud/Run_*/*.npz"), seed=0)
    batch = collate([ds[i] for i in range(8)])
    batch["track_points"]   # (B, T, 33, 3)  x,y,z along each track, inner -> outer
    batch["track_feats"]    # (B, T, F)  track_features (default pt only; eta, phi, charge,
                            #            chi2, ndof come from track_info)
    batch["track_mask"]     # (B, T) True where a real track is
    batch["hit_feats"]      # (B, H, 5)  x,y,z,energy,time; hit_det = index into hit_keys
    batch["seg_feats"]      # (B, S, 6)  segment end points x1,y1,z1,x2,y2,z2; seg_det 0=CSC 1=DT
    batch["rpc_feats"]      # (B, R, 3)  RPC hit x,y,z
    batch["track_is_muon"]  # (B, T) per-track label (False on padding)
"""
import numpy as np

TRACK_POINT_FEATURES = ("x", "y", "z")                  # per point, in Bezier order
TRACK_FEATURES = ("pt",)                                # per track (default)
# Per-track features that can be chosen (track_features=...): pt from tracks, the rest from
# track_info (row i = track_id i). chi2 / ndof carry the electron brem signature.
TRACK_INFO_FEATURES = ("pt", "eta", "phi", "charge", "chi2", "ndof")
HIT_FEATURES = ("x", "y", "z", "energy", "time")       # per hit (eta, phi follow from x, y, z)
# Default hit_keys; hit_det = position in the hit_keys used. ECAL points are on the crystal
# front face, HCAL points at the cell centre (agent/pointcloud.py).
HIT_KEYS = ("ecal_eb", "ecal_ee", "hcal_hb", "hcal_he", "hcal_hf", "hcal_ho")
SEG_FEATURES = ("x1", "y1", "z1", "x2", "y2", "z2")     # per segment
SEG_KEYS = ("csc_segments", "dt_segments")              # seg_det = position in this tuple
RPC_FEATURES = ("x", "y", "z")                          # per RPC hit
POINTS_PER_TRACK = 33
MATCH_TOL = 1e-3                                        # m, track <-> tracker muon end points

_DEFAULT_COLUMNS = {
    "tracks": ["x", "y", "z", "pt", "track_id"],
    "global_muon": ["x", "y", "z", "pt", "muon_id"],
    "muon_tracks": ["x", "y", "z", "pt", "track_id"],
    "elec_tracks": ["x", "y", "z", "pt", "track_id"],
    "track_info": ["pt", "eta", "phi", "charge", "chi2", "ndof", "track_id"],
    "ecal_eb": ["x", "y", "z", "energy", "eta", "phi", "time", "detid"],
    "ecal_ee": ["x", "y", "z", "energy", "eta", "phi", "time", "detid"],
    **{k: ["x", "y", "z", "energy", "eta", "phi", "time", "detid"]
       for k in ("hcal_hb", "hcal_he", "hcal_hf", "hcal_ho")},
    "csc_segments": ["x1", "y1", "z1", "x2", "y2", "z2", "endcap", "station", "ring", "chamber", "detid"],
    "dt_segments": ["x1", "y1", "z1", "x2", "y2", "z2", "wheel", "station", "sector", "detid"],
    "rpc_hits": ["x", "y", "z", "region", "station", "sector", "layer", "detid"],
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


def tracks(npz, key="tracks", features=TRACK_FEATURES):
    """-> points (T, 33, 3) float32, features (T, len(features)) float32.
    A feature not among the key's columns is looked up in track_info by track_id."""
    arr, c = _array(npz, key)
    groups = _group_by_id(arr, c["track_id"])
    bad = [len(g) for g in groups if len(g) != POINTS_PER_TRACK]
    if bad:
        raise ValueError(f"tracks with {sorted(set(bad))} points, expected {POINTS_PER_TRACK}")
    p_cols = [c[f] for f in TRACK_POINT_FEATURES]
    points = np.array([g[:, p_cols] for g in groups], dtype=np.float32).reshape(
        -1, POINTS_PER_TRACK, len(p_cols))
    ids = np.array([g[0, c["track_id"]] for g in groups], dtype=np.int64)
    feats = np.zeros((len(groups), len(features)), dtype=np.float32)
    info = None
    for j, f in enumerate(features):
        if f in c:
            feats[:, j] = [g[0, c[f]] for g in groups]
            continue
        if info is None:
            info, ic = _array(npz, "track_info")
            rows = {int(t): i for i, t in enumerate(info[:, ic["track_id"]])}
            missing = [int(t) for t in ids if int(t) not in rows]
            if missing:
                raise ValueError(f"{len(missing)} tracks have no track_info row "
                                 f"(npz made before track_info? remake with --steps pointcloud)")
            sel = info[[rows[int(t)] for t in ids]] if len(ids) else info[:0]
        feats[:, j] = sel[:, ic[f]]
    return points, feats


def _stack(npz, keys, features):
    """Rows of several keys, selected columns -> float32 (N, F), source index (N,) int64."""
    feats, det = [], []
    for d, key in enumerate(keys):
        arr, c = _array(npz, key)
        feats.append(arr[:, [c[f] for f in features]].astype(np.float32))
        det.append(np.full(len(arr), d, dtype=np.int64))
    return np.concatenate(feats), np.concatenate(det)


def hits(npz, keys=HIT_KEYS):
    """-> features (H, len(HIT_FEATURES)) float32, detector (H,) int64 (index into keys)"""
    return _stack(npz, keys, HIT_FEATURES)


def segments(npz):
    """-> end points (S, 6) float32, detector (S,) int64 (index into SEG_KEYS)"""
    return _stack(npz, SEG_KEYS, SEG_FEATURES)


def rpc_hits(npz):
    """-> (R, 3) float32"""
    return _stack(npz, ("rpc_hits",), RPC_FEATURES)[0]


def muons(npz, pt_min):
    """pT of the global muons above pt_min (label material, never an input)."""
    arr, c = _array(npz, "global_muon")
    pts = np.array([g[0, c["pt"]] for g in _group_by_id(arr, c["muon_id"])], dtype=np.float32)
    return pts[pts > pt_min]


def match_tracks(track_points, muon_points, tol=MATCH_TOL):
    """For each track, distance (m) to the closest muon track, taken as the larger of the
    two end-point distances; inf when there is no muon track. -> (T,) float32, (T,) index"""
    if not len(track_points) or not len(muon_points):
        return np.full(len(track_points), np.inf, dtype=np.float32), np.full(len(track_points), -1)
    ends_t = track_points[:, [0, -1]]                     # (T, 2, 3)
    ends_m = muon_points[:, [0, -1]]                      # (M, 2, 3)
    d = np.linalg.norm(ends_t[:, None] - ends_m[None], axis=-1).max(axis=-1)   # (T, M)
    j = d.argmin(axis=1)
    return d[np.arange(len(d)), j], j


def track_labels(npz, track_points, pt_min, tol=MATCH_TOL):
    """-> (T,) bool: track matched to a tracker muon with pT > pt_min (label, never an input)."""
    muon_points, muon_feats = tracks(npz, "muon_tracks")
    keep = muon_feats[:, TRACK_FEATURES.index("pt")] > pt_min
    dist, _ = match_tracks(track_points, muon_points[keep], tol)
    return dist < tol


def event(path, rng=None, pt_min=10.0, hit_keys=HIT_KEYS, match_tol=MATCH_TOL,
          track_features=TRACK_FEATURES):
    """One npz -> dict of inputs and label. rng: shuffle tracks and hits (None keeps file order).
    hit_keys: calorimeter parts to use, e.g. ("ecal_eb", "ecal_ee") for ECAL only.
    match_tol: end-point distance (m) for track_is_muon.
    track_features: per-track columns of track_feats, any of TRACK_INFO_FEATURES."""
    with np.load(path) as npz:
        track_points, track_feats = tracks(npz, "tracks", track_features)
        track_is_muon = track_labels(npz, track_points, pt_min, match_tol)
        hit_feats, hit_det = hits(npz, hit_keys)
        seg_feats, seg_det = segments(npz)
        rpc_feats = rpc_hits(npz)
        muon_pt = muons(npz, pt_min)
        ident = tuple(int(npz[k][0]) for k in ("run", "lumi", "event") if k in npz.files)
    if rng is not None:
        t = rng.permutation(len(track_points))
        track_points, track_feats, track_is_muon = track_points[t], track_feats[t], track_is_muon[t]
        h = rng.permutation(len(hit_feats))
        hit_feats, hit_det = hit_feats[h], hit_det[h]
        g = rng.permutation(len(seg_feats))
        seg_feats, seg_det = seg_feats[g], seg_det[g]
        rpc_feats = rpc_feats[rng.permutation(len(rpc_feats))]
    return {
        "track_points": track_points, "track_feats": track_feats,
        "hit_feats": hit_feats, "hit_det": hit_det,
        "seg_feats": seg_feats, "seg_det": seg_det, "rpc_feats": rpc_feats,
        "has_muon": np.int64(len(muon_pt) > 0), "n_muons": np.int64(len(muon_pt)),
        "track_is_muon": track_is_muon,
        "id": ident, "path": str(path),
    }


class NpzEventDataset:
    """Sequence of events; works as a torch Dataset (len + getitem) with collate_fn=collate.

    Tracks and hits are reshuffled on every access. With DataLoader workers, give each
    worker its own seed (worker_init_fn) so the workers do not repeat the same order.
    """

    def __init__(self, paths, seed=None, shuffle=True, pt_min=10.0, hit_keys=HIT_KEYS,
                 match_tol=MATCH_TOL, track_features=TRACK_FEATURES):
        self.paths = sorted(str(p) for p in paths)
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.pt_min = pt_min
        self.hit_keys = tuple(hit_keys)
        self.match_tol = match_tol
        self.track_features = tuple(track_features)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return event(self.paths[i], self.rng if self.shuffle else None, self.pt_min, self.hit_keys,
                     self.match_tol, self.track_features)


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
    seg_feats, seg_mask = _pad([e["seg_feats"] for e in events])
    seg_det, _ = _pad([e["seg_det"] for e in events])
    rpc_feats, rpc_mask = _pad([e["rpc_feats"] for e in events])
    track_is_muon, _ = _pad([e["track_is_muon"] for e in events], fill=False)
    return {
        "track_points": track_points, "track_feats": track_feats, "track_mask": track_mask,
        "hit_feats": hit_feats, "hit_det": hit_det, "hit_mask": hit_mask,
        "seg_feats": seg_feats, "seg_det": seg_det, "seg_mask": seg_mask,
        "rpc_feats": rpc_feats, "rpc_mask": rpc_mask, "track_is_muon": track_is_muon,
        "has_muon": np.array([e["has_muon"] for e in events]),
        "n_muons": np.array([e["n_muons"] for e in events]),
    }
