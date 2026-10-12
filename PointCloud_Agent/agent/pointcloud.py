"""
One iSpy event (ig JSON) -> point cloud arrays.

The geometry logic is unchanged from
ig/ig_and_pointcloud/DoubleElectron_Run2012C_0/code/generate_pointcloud_260911.py:
  - tracks / GSF electrons / tracker & standalone muons: cubic Bezier from Extras_V1 (33 points),
    same construction as iSpy (ispy-webgl/js/objects-draw.js)
  - global muons: Points_V1 associated through MuonGlobalPoints_V1
  - ECAL EB/EE: intersection of the front-face diagonals; HCAL HB: box centre
  - muon chambers: box centre
Changes:
  - fields are looked up by name from the ig "Types" block (robust across ISpy versions)
  - missing collections/associations give empty arrays instead of KeyError
  - vertices: PrimaryVertices_V1 (Run2/Run3 ISpy) or Vertices_V1 (Run1), plus SecondaryVertices_V1
"""

import json
import re
from collections import OrderedDict

import numpy as np

# Rechit energy cuts (GeV), matching what iSpy applies when drawing
MIN_E_EB = 0.25
MIN_E_EE = 0.5
MIN_E_HB = 0.0

BEZIER_POINTS = 32   # -> 33 points per track

# Fallback field positions when an event has no "Types" entry for a collection
_DEFAULT_FIELDS = {
    "Tracks_V3": ["pos", "dir", "pt", "phi", "eta", "charge", "chi2", "ndof"],
    "Extras_V1": ["pos_1", "dir_1", "pos_2", "dir_2"],
    "GsfElectrons_V2": ["pt", "eta", "phi", "charge", "pos", "dir"],
    "TrackerMuons_V2": ["pt", "charge", "pos", "phi", "eta"],
    "GlobalMuons_V1": ["pt", "charge", "rp", "phi", "eta", "calo_energy"],
    "StandaloneMuons_V2": ["pt", "charge", "pos", "phi", "eta", "calo_energy"],
    "Points_V1": ["pos"],
    "MuonChambers_V1": ["detid", "front_1", "front_2", "front_3", "front_4",
                        "back_1", "back_2", "back_3", "back_4"],
    "Vertices_V1": ["isValid", "isFake", "pos", "xError", "yError", "zError", "chi2", "ndof"],
    "PrimaryVertices_V1": ["isValid", "isFake", "pos", "xError", "yError", "zError", "chi2", "ndof"],
    "SecondaryVertices_V1": ["isValid", "isFake", "pos", "xError", "yError", "zError", "chi2", "ndof"],
}
_RECHIT_FIELDS = ["energy", "eta", "phi", "time", "detid", "front_1", "front_2", "front_3", "front_4",
                  "back_1", "back_2", "back_3", "back_4"]

# Column meaning of each output array (also stored in the npz as "<key>_columns")
COLUMNS = OrderedDict([
    ("tracks",             ["x", "y", "z", "pt", "track_id"]),
    ("elec_tracks",        ["x", "y", "z", "pt", "track_id"]),
    ("muon_tracks",        ["x", "y", "z", "pt", "track_id"]),
    ("global_muon",        ["x", "y", "z", "pt", "muon_id"]),
    ("standalone_muon",    ["x", "y", "z", "pt", "muon_id"]),
    ("muon_chambers",      ["x", "y", "z", "detid"]),
    ("vertices",           ["x", "y", "z"]),
    ("secondary_vertices", ["x", "y", "z"]),
    ("ecal_eb",            ["x", "y", "z", "energy", "eta", "phi", "time", "detid"]),
    ("ecal_ee",            ["x", "y", "z", "energy", "eta", "phi", "time", "detid"]),
    ("hcal_hb",            ["x", "y", "z", "energy", "eta", "phi", "time", "detid"]),
])


# ════════════════════════════════════════════════════════
#  ig access helpers
# ════════════════════════════════════════════════════════

def load_ig_text(text):
    """ig JSON uses bare nan/inf; map them to null before parsing."""
    text = re.sub(r'-?\b(?:nan|NaN|Infinity|inf)\b', 'null', text)
    return json.loads(text)


def load_ig(path):
    with open(path, "rb") as f:
        return load_ig_text(f.read().decode("utf-8", errors="replace"))


def _collection(data, name):
    return data.get("Collections", {}).get(name, []) or []


def _assoc(data, name):
    return data.get("Associations", {}).get(name, []) or []


def _fields(data, name):
    types = data.get("Types", {}).get(name)
    if types:
        return {f[0]: i for i, f in enumerate(types)}
    defaults = _DEFAULT_FIELDS.get(name) or (_RECHIT_FIELDS if name.endswith("RecHits_V2") else [])
    return {f: i for i, f in enumerate(defaults)}


def event_id(data):
    """(run, event) from Event_V2 / Event_V3."""
    for name in ("Event_V3", "Event_V2", "Event_V1"):
        rows = _collection(data, name)
        if rows:
            f = _fields(data, name)
            return int(rows[0][f.get("run", 0)]), int(rows[0][f.get("event", 1)])
    return None, None


# ════════════════════════════════════════════════════════
#  geometry
# ════════════════════════════════════════════════════════

def cubic_bezier(p1, p3, p4, p2, n_points=BEZIER_POINTS):
    """iSpy CubicBezierCurve3(p1,p3,p4,p2).getPoints(n_points) -> (n_points+1, 3)"""
    t = np.linspace(0, 1, n_points + 1)[:, np.newaxis]
    pts = ((1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * p3 + 3 * (1 - t) * t ** 2 * p4 + t ** 3 * p2)
    return pts.astype(np.float32)


def bezier_from_extra(extra, fields, n_points=BEZIER_POINTS):
    p1 = np.array(extra[fields["pos_1"]], dtype=np.float64)
    d1 = np.array(extra[fields["dir_1"]], dtype=np.float64)
    p2 = np.array(extra[fields["pos_2"]], dtype=np.float64)
    d2 = np.array(extra[fields["dir_2"]], dtype=np.float64)
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    scale = np.linalg.norm(p2 - p1) * 0.25
    return cubic_bezier(p1, p1 + scale * d1, p2 - scale * d2, p2, n_points)


def order_quad(corners):
    """Sort 4 face corners by angle around the face centre."""
    corners = np.asarray(corners, dtype=np.float64)
    c = corners.mean(axis=0)
    n = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    nn = np.linalg.norm(n)
    if nn < 1e-15:
        return corners
    n /= nn
    u = corners[0] - c
    u -= n * (u @ n)
    un = np.linalg.norm(u)
    if un < 1e-15:
        return corners
    u /= un
    v = np.cross(n, u)
    rel = corners - c
    return corners[np.argsort(np.arctan2(rel @ v, rel @ u))]


def quad_diag_point(corners):
    """Intersection of the two diagonals of a face (falls back to the corner mean)."""
    q = order_quad(corners)
    p1, p2, p3, p4 = q
    d1, d2, r = p3 - p1, p4 - p2, p2 - p1
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    det = a * c - b * b
    if abs(det) < 1e-18:
        return q.mean(axis=0)
    t = (c * (r @ d1) - b * (r @ d2)) / det
    s = (b * (r @ d1) - a * (r @ d2)) / det
    return ((p1 + t * d1) + (p2 + s * d2)) / 2.0


# ════════════════════════════════════════════════════════
#  per-collection extraction
# ════════════════════════════════════════════════════════

def _empty(key):
    return np.zeros((0, len(COLUMNS[key])), dtype=np.float32)


def _bezier_objects(data, coll_name, assoc_name, key):
    """Objects drawn as Bezier curves through Extras_V1 -> (N*33, 5) [x,y,z,pt,id]"""
    objs = _collection(data, coll_name)
    extras = _collection(data, "Extras_V1")
    assoc = _assoc(data, assoc_name)
    if not objs or not extras or not assoc:
        return _empty(key)
    pt_i = _fields(data, coll_name)["pt"]
    ef = _fields(data, "Extras_V1")
    out = []
    for obj_id, a in enumerate(assoc):
        oi, ei = a[0][1], a[1][1]
        pts = bezier_from_extra(extras[ei], ef)
        n = len(pts)
        out.append(np.column_stack([pts,
                                    np.full(n, objs[oi][pt_i], dtype=np.float32),
                                    np.full(n, obj_id, dtype=np.float32)]))
    return np.vstack(out).astype(np.float32)


def tracks(data):
    return _bezier_objects(data, "Tracks_V3", "TrackExtras_V1", "tracks")


def electron_tracks(data):
    return _bezier_objects(data, "GsfElectrons_V2", "GsfElectronExtras_V1", "elec_tracks")


def tracker_muon_tracks(data):
    return _bezier_objects(data, "TrackerMuons_V2", "MuonTrackerExtras_V1", "muon_tracks")


def standalone_muons(data):
    return _bezier_objects(data, "StandaloneMuons_V2", "MuonTrackExtras_V1", "standalone_muon")


def global_muons(data):
    """GlobalMuons_V1 -> Points_V1 via MuonGlobalPoints_V1 -> (N, 5) [x,y,z,pt,muon_id]"""
    points = _collection(data, "Points_V1")
    muons = _collection(data, "GlobalMuons_V1")
    assoc = _assoc(data, "MuonGlobalPoints_V1")
    if not points or not muons or not assoc:
        return _empty("global_muon")
    pos_i = _fields(data, "Points_V1")["pos"]
    pt_i = _fields(data, "GlobalMuons_V1")["pt"]
    per_muon = OrderedDict()
    for a in assoc:
        per_muon.setdefault(a[0][1], []).append(points[a[1][1]][pos_i])
    out = []
    for muon_id, (mi, pts) in enumerate(per_muon.items()):
        pts = np.array(pts, dtype=np.float32)
        n = len(pts)
        out.append(np.column_stack([pts,
                                    np.full(n, muons[mi][pt_i], dtype=np.float32),
                                    np.full(n, muon_id, dtype=np.float32)]))
    return np.vstack(out).astype(np.float32)


def muon_chambers(data):
    """MuonChambers_V1 -> (N, 4) [x,y,z,detid], box centre"""
    rows = _collection(data, "MuonChambers_V1")
    if not rows:
        return _empty("muon_chambers")
    f = _fields(data, "MuonChambers_V1")
    corner_i = [f[k] for k in ("front_1", "front_2", "front_3", "front_4",
                               "back_1", "back_2", "back_3", "back_4")]
    out = []
    for ch in rows:
        c = np.mean([ch[i] for i in corner_i], axis=0)
        out.append([c[0], c[1], c[2], float(ch[f["detid"]])])
    return np.array(out, dtype=np.float64)


def _vertices(data, names, key):
    for name in names:
        rows = _collection(data, name)
        if rows:
            pos_i = _fields(data, name)["pos"]
            return np.array([r[pos_i] for r in rows], dtype=np.float32)
    return _empty(key)


def primary_vertices(data):
    return _vertices(data, ("PrimaryVertices_V1", "Vertices_V1"), "vertices")


def secondary_vertices(data):
    return _vertices(data, ("SecondaryVertices_V1",), "secondary_vertices")


def _rechits(data, coll_name, min_energy, front_face, key):
    """rechits -> (N, 8) [x,y,z,energy,eta,phi,time,detid]"""
    rows = _collection(data, coll_name)
    if not rows:
        return _empty(key)
    f = _fields(data, coll_name)
    front = [f[k] for k in ("front_1", "front_2", "front_3", "front_4")]
    back = [f[k] for k in ("back_1", "back_2", "back_3", "back_4")]
    out = []
    for h in rows:
        e = h[f["energy"]]
        if e is None or (e < min_energy if min_energy > 0 else e <= min_energy):
            continue
        if front_face:
            p = quad_diag_point([h[i] for i in front])
        else:
            p = np.mean([h[i] for i in front + back], axis=0)
        out.append([p[0], p[1], p[2], e, h[f["eta"]], h[f["phi"]], h[f["time"]], float(h[f["detid"]])])
    return np.array(out, dtype=np.float64) if out else _empty(key)


def ecal_eb(data):
    return _rechits(data, "EBRecHits_V2", MIN_E_EB, True, "ecal_eb")


def ecal_ee(data):
    return _rechits(data, "EERecHits_V2", MIN_E_EE, True, "ecal_ee")


def hcal_hb(data):
    return _rechits(data, "HBRecHits_V2", MIN_E_HB, False, "hcal_hb")


_EXTRACTORS = OrderedDict([
    ("tracks", tracks),
    ("elec_tracks", electron_tracks),
    ("muon_tracks", tracker_muon_tracks),
    ("global_muon", global_muons),
    ("standalone_muon", standalone_muons),
    ("muon_chambers", muon_chambers),
    ("vertices", primary_vertices),
    ("secondary_vertices", secondary_vertices),
    ("ecal_eb", ecal_eb),
    ("ecal_ee", ecal_ee),
    ("hcal_hb", hcal_hb),
])


def build(data):
    """ig event dict -> OrderedDict of arrays, keys as in COLUMNS."""
    return OrderedDict((key, fn(data)) for key, fn in _EXTRACTORS.items())


def save_npz(path, arrays, run, event):
    """Same keys as the 260911 output, plus secondary_vertices and <key>_columns."""
    extra = {f"{k}_columns": np.array(v) for k, v in COLUMNS.items()}
    np.savez(path, run=np.array([run]), event=np.array([event]), **arrays, **extra)


def object_counts(arrays):
    """Number of distinct objects per key (ids for curves, rows otherwise)."""
    counts = {}
    for key, arr in arrays.items():
        if len(arr) and COLUMNS[key][-1] in ("track_id", "muon_id"):
            counts[key] = int(len(np.unique(arr[:, -1])))
        else:
            counts[key] = int(len(arr))
    return counts
