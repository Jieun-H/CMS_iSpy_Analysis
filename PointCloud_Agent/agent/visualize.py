"""
Point cloud arrays -> standalone HTML page (plotly 3D scatter).

Look and feel taken from generate_pointcloud_260911.py: same colours, marker size,
dark/light toggle and HTML sidebar legend. The GLTF overlay is gone (GLTF had to be
exported by hand from the iSpy web app), so only the ig points are drawn.
"""

import html
import json

import numpy as np
import plotly.graph_objects as go

POINT_SIZE = 0.5
SYMBOL = "circle"

# (key, legend name, colour, hover label of column 3, hover label of column 4)
LAYERS = [
    ("tracks",             "Track",              "lime",       "pT", "track_id"),
    ("elec_tracks",        "Electron",           "cyan",       "pT", "track_id"),
    ("muon_tracks",        "TrackerMuon",        "red",        "pT", "muon_id"),
    ("global_muon",        "GlobalMuon",         "orange",     "pT", "muon_id"),
    ("standalone_muon",    "StandaloneMuon",     "hotpink",    "pT", "muon_id"),
    ("muon_chambers",      "MuonChamber",        "white",      "detid", None),
    ("vertices",           "PrimaryVertex",      "yellow",     None, None),
    ("secondary_vertices", "SecondaryVertex",    "gold",       None, None),
    ("ecal_eb",            "ECAL EB",            "dodgerblue", "E", None),
    ("ecal_ee",            "ECAL EE",            "royalblue",  "E", None),
    ("hcal_hb",            "HCAL HB",            "turquoise",  "E", None),
]
_COLOR = {k: c for k, _, c, _, _ in LAYERS}

DARK_BG, DARK_FG, DARK_GRID, DARK_ZERO = "#000000", "#ffffff", "#333333", "#555555"
LIGHT_BG, LIGHT_FG, LIGHT_GRID, LIGHT_ZERO = "#ffffff", "#000000", "#dddddd", "#aaaaaa"


def _flatten_theme(bg, fg, grid, zero):
    """dot-notation layout for Plotly.relayout (theme buttons)"""
    flat = {"paper_bgcolor": bg, "plot_bgcolor": bg, "font.color": fg, "legend.bgcolor": bg}
    for ax in ("xaxis", "yaxis", "zaxis"):
        flat[f"scene.{ax}.backgroundcolor"] = bg
        flat[f"scene.{ax}.gridcolor"] = grid
        flat[f"scene.{ax}.zerolinecolor"] = zero
        flat[f"scene.{ax}.color"] = fg
    return flat


def _nested_theme(bg, fg, grid, zero):
    axis = dict(backgroundcolor=bg, gridcolor=grid, zerolinecolor=zero, color=fg)
    return dict(paper_bgcolor=bg, plot_bgcolor=bg, font=dict(color=fg), legend=dict(bgcolor=bg),
                scene=dict(xaxis=axis, yaxis=axis, zaxis=axis))


DARK_THEME = _nested_theme(DARK_BG, DARK_FG, DARK_GRID, DARK_ZERO)
DARK_THEME_FLAT = _flatten_theme(DARK_BG, DARK_FG, DARK_GRID, DARK_ZERO)
LIGHT_THEME_FLAT = _flatten_theme(LIGHT_BG, LIGHT_FG, LIGHT_GRID, LIGHT_ZERO)


def make_figure(arrays):
    fig = go.Figure()
    for key, name, color, c3, c4 in LAYERS:
        arr = arrays.get(key)
        if arr is None or len(arr) == 0:
            continue
        hover = "(%{x:.3f}, %{y:.3f}, %{z:.3f})"
        custom = None
        if c3 and c4:
            custom = arr[:, 3:5]
            hover += f"<br>{c3}: %{{customdata[0]:.2f}}<br>{c4}: %{{customdata[1]:.0f}}"
        elif c3:
            custom = arr[:, 3:4]
            fmt = ".0f" if c3 == "detid" else ".3f"
            hover += f"<br>{c3}: %{{customdata[0]:{fmt}}}"
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="markers", name=name,
            marker=dict(size=POINT_SIZE, color=color, symbol=SYMBOL, opacity=0.8),
            customdata=custom, hovertemplate=hover + f"<extra>{name}</extra>"))

    # The legend is an HTML sidebar (see render_page), not plotly's own.
    fig.update_layout(
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)", aspectmode="data"),
        showlegend=False, height=820, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_layout(DARK_THEME)
    return fig


PAGE_CSS = """
:root {
  --bg: #000000; --card: #111111; --border: #262626;
  --fg: #ffffff; --muted: #9a9a9a; --accent: #4da3ff;
}
:root[data-theme="light"], body[data-theme="light"] {
  --bg: #ffffff; --card: #f7f7f8; --border: #e4e4e7;
  --fg: #0a0a0a; --muted: #6b7280; --accent: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.page { max-width: 1400px; margin: 0 auto; padding: 20px 24px 40px; }
h1.page-title { font-size: 20px; font-weight: 600; margin: 0 0 2px; }
.page-subtitle { color: var(--muted); font-size: 13px; margin: 0 0 16px; word-break: break-all; }
.stats { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.stat-tile {
  display: flex; align-items: center; gap: 8px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 8px 14px;
}
.stat-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.stat-label { color: var(--muted); font-size: 12px; }
.stat-value { font-weight: 600; font-size: 16px; }
.layout { display: flex; gap: 16px; align-items: flex-start; }
.chart-col { flex: 1 1 auto; min-width: 0; }
.sidebar { width: 230px; flex: 0 0 230px; display: flex; flex-direction: column; gap: 10px; }
.toolbar { display: flex; gap: 8px; }
.theme-btn {
  flex: 1; border: 1px solid var(--border); background: var(--card); color: var(--fg);
  padding: 6px 10px; border-radius: 8px; cursor: pointer; font-size: 13px;
}
.theme-btn.active { border-color: var(--accent); color: var(--accent); }
.chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 8px; }
.legend-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 8px; max-height: 780px; overflow-y: auto;
}
.legend-row {
  display: flex; align-items: center; gap: 8px; padding: 5px 6px;
  border-radius: 6px; cursor: pointer; font-size: 12.5px; user-select: none;
}
.legend-row:hover { background: rgba(128,128,128,0.15); }
.legend-row.hidden { opacity: 0.35; text-decoration: line-through; }
.legend-swatch { flex: 0 0 auto; width: 12px; height: 12px; border-radius: 50%; background: currentColor; }
@media (max-width: 800px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; flex: 0 0 auto; }
}
"""

_JS_TEMPLATE = """
<script>
const DARK_FLAT  = __DARK_JSON__;
const LIGHT_FLAT = __LIGHT_JSON__;
function setTheme(name) {
  document.body.setAttribute('data-theme', name);
  document.getElementById('btn-dark').classList.toggle('active', name === 'dark');
  document.getElementById('btn-light').classList.toggle('active', name === 'light');
  var gd = document.getElementById('ispy-chart');
  if (gd && window.Plotly) { Plotly.relayout(gd, name === 'dark' ? DARK_FLAT : LIGHT_FLAT); }
}
setTheme('dark');
function toggleTrace(idx, el) {
  var gd = document.getElementById('ispy-chart');
  var hidden = el.classList.toggle('hidden');
  if (gd && window.Plotly) { Plotly.restyle(gd, {visible: hidden ? false : true}, [idx]); }
}
// Resize once the flex layout has settled so the WebGL canvas does not cover the sidebar.
function resizeChart() {
  var gd = document.getElementById('ispy-chart');
  if (gd && window.Plotly) { Plotly.Plots.resize(gd); }
}
window.addEventListener('load', function() { resizeChart(); setTimeout(resizeChart, 200); });
window.addEventListener('resize', resizeChart);
</script>
"""


def render_page(arrays, counts, run, event, subtitle):
    fig = make_figure(arrays)
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="ispy-chart",
                            config={"responsive": True})

    legend_rows = "\n".join(
        f'<div class="legend-row" onclick="toggleTrace({i}, this)">'
        f'<span class="legend-swatch" style="color:{tr.marker.color}"></span>'
        f'<span>{html.escape(tr.name)}</span></div>'
        for i, tr in enumerate(fig.data))

    n_muons = counts["muon_tracks"] + counts["global_muon"] + counts["standalone_muon"]
    tiles_data = [
        ("Tracks", counts["tracks"], _COLOR["tracks"], None),
        ("Electrons", counts["elec_tracks"], _COLOR["elec_tracks"], None),
        ("Muons", n_muons, _COLOR["muon_tracks"],
         f"TrackerMuon {counts['muon_tracks']} · GlobalMuon {counts['global_muon']} · "
         f"StandaloneMuon {counts['standalone_muon']}"),
        ("Vertices", counts["vertices"], _COLOR["vertices"],
         f"Primary {counts['vertices']} · Secondary {counts['secondary_vertices']}"),
        ("ECAL hits", counts["ecal_eb"] + counts["ecal_ee"], _COLOR["ecal_eb"], None),
        ("HCAL HB hits", counts["hcal_hb"], _COLOR["hcal_hb"], None),
    ]
    tiles = "\n".join(
        f'<div class="stat-tile"{f" title={json.dumps(title)}" if title else ""}>'
        f'<span class="stat-dot" style="background:{color}"></span>'
        f'<span class="stat-label">{label}</span><span class="stat-value">{value}</span></div>'
        for label, value, color, title in tiles_data)

    js_block = (_JS_TEMPLATE.replace("__DARK_JSON__", json.dumps(DARK_THEME_FLAT))
                .replace("__LIGHT_JSON__", json.dumps(LIGHT_THEME_FLAT)))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run {run} · Event {event}</title>
<style>{PAGE_CSS}</style>
</head>
<body data-theme="dark">
<div class="page">
  <h1 class="page-title">Run {run} &middot; Event {event}</h1>
  <p class="page-subtitle">{html.escape(subtitle)}</p>
  <div class="stats">
    {tiles}
  </div>
  <div class="layout">
    <div class="chart-col"><div class="chart-card">{plot_html}</div></div>
    <aside class="sidebar">
      <div class="toolbar">
        <button class="theme-btn" id="btn-dark" onclick="setTheme('dark')">&#9899; Dark</button>
        <button class="theme-btn" id="btn-light" onclick="setTheme('light')">&#9898; Light</button>
      </div>
      <div class="legend-card">
        {legend_rows}
      </div>
    </aside>
  </div>
</div>
{js_block}
</body>
</html>
"""
