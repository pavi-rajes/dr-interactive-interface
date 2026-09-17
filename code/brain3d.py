"""Reusable plotly 3D mouse-brain figure with unit dots.

Coordinate convention
---------------------
Meshes and units are stored in Allen CCF micrometres with atlas axis order
(ap, dv, ml): ap increases anterior -> posterior, dv increases superior -> inferior,
ml increases right -> left (midline ~5700 um). For display, `to_plot_coords` maps
    x = ml          (right hemisphere at low x)
    y = -ap         (anterior at high y, i.e. towards the viewer in the default camera)
    z = -dv         (dorsal up)
The same mapping is applied to mesh vertices and unit dots so they stay aligned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import config  # when imported from code/
    _DEFAULT_MESH_DIR = config.PRECOMPUTED_DIR / "ccf_meshes"
except ImportError:  # pragma: no cover
    _DEFAULT_MESH_DIR = Path(__file__).resolve().parents[1] / "data" / "precomputed" / "ccf_meshes"


def load_mesh_index(mesh_dir: Path = _DEFAULT_MESH_DIR) -> dict:
    return json.loads((Path(mesh_dir) / "index.json").read_text())


def load_mesh(acronym: str, mesh_dir: Path = _DEFAULT_MESH_DIR, prefer_decimated: bool = True) -> tuple[np.ndarray, np.ndarray]:
    mesh_dir = Path(mesh_dir)
    path = mesh_dir / f"{acronym}.npz"
    if acronym == "root" and prefer_decimated and (mesh_dir / "root_decimated.npz").exists():
        path = mesh_dir / "root_decimated.npz"
    z = np.load(path)
    return z["vertices"], z["faces"]


def to_plot_coords(ap, dv, ml):
    ap, dv, ml = np.asarray(ap, dtype=float), np.asarray(dv, dtype=float), np.asarray(ml, dtype=float)
    return ml, -ap, -dv


def _mesh_trace(verts, faces, color, opacity, name, hover=False) -> go.Mesh3d:
    x, y, z = to_plot_coords(verts[:, 0], verts[:, 1], verts[:, 2])
    return go.Mesh3d(x=x, y=y, z=z, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color, opacity=opacity,
                     flatshading=True, name=name, showlegend=name != "root", hoverinfo="name" if hover else "skip",
                     lighting=dict(ambient=0.6, diffuse=0.6, specular=0.05), showscale=False)


def brain_figure(units: pd.DataFrame, color_col: str | None = None, hover_cols=("unit_id", "structure", "session_id"),
                 selected_unit: tuple[str, int] | None = None, region_meshes=(), mesh_opacity: float = 0.08,
                 region_opacity: float = 0.15, colorscale: str = "RdBu", color_range: tuple[float, float] | None = None,
                 marker_size: float = 3, title: str | None = None, mesh_dir: Path = _DEFAULT_MESH_DIR,
                 colorbar_title: str | None = None) -> go.Figure:
    fig = go.Figure()
    rv, rf = load_mesh("root", mesh_dir)
    fig.add_trace(_mesh_trace(rv, rf, "#9a9a9a", mesh_opacity, "root"))
    index = load_mesh_index(mesh_dir)["structures"]
    for acr in region_meshes:
        if acr not in index:
            continue
        v, f = load_mesh(acr, mesh_dir)
        rgb = index[acr]["rgb"]
        fig.add_trace(_mesh_trace(v, f, f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", region_opacity, acr, hover=True))

    u = units.dropna(subset=["ccf_ap", "ccf_dv", "ccf_ml"])
    n_dropped = len(units) - len(u)
    x, y, z = to_plot_coords(u["ccf_ap"], u["ccf_dv"], u["ccf_ml"])
    hover_cols = [c for c in hover_cols if c in u.columns]
    hover = "<br>".join(f"{c}: %{{customdata[{i + 2}]}}" for i, c in enumerate(hover_cols))
    if color_col:
        hover += f"<br>{color_col}: %{{marker.color:.3g}}"
    custom = np.column_stack([u["session_id"].astype(str).to_numpy(), u["unit_index"].to_numpy()]
                             + [u[c].astype(str).to_numpy() for c in hover_cols])
    marker = dict(size=marker_size, opacity=0.85)
    if color_col:
        marker.update(color=u[color_col].to_numpy(), colorscale=colorscale, showscale=True,
                      colorbar=dict(title=colorbar_title or color_col, thickness=12, len=0.5, x=0.98))
        if color_range:
            marker.update(cmin=color_range[0], cmax=color_range[1])
    else:
        marker.update(color="#1f77b4")
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker, customdata=custom,
                               hovertemplate=hover + "<extra></extra>", name=f"units (n={len(u)})"))
    sx, sy, sz = [], [], []
    if selected_unit is not None:
        sid, ui = selected_unit
        m = (u["session_id"] == sid) & (u["unit_index"] == ui)
        if m.any():
            sx, sy, sz = (list(v) for v in to_plot_coords(u.loc[m, "ccf_ap"], u.loc[m, "ccf_dv"], u.loc[m, "ccf_ml"]))
    # always present (possibly empty) so callers can move the highlight without rebuilding the figure
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers", name="selected unit",
                               marker=dict(size=marker_size * 3, color="#ffd400", line=dict(color="black", width=2)),
                               hoverinfo="skip", showlegend=True))
    axis = dict(showticklabels=False, showgrid=False, zeroline=False, showbackground=False)
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        scene=dict(aspectmode="data", xaxis={**axis, "title": "ML"}, yaxis={**axis, "title": "AP"}, zaxis={**axis, "title": "DV"},
                   camera=dict(eye=dict(x=1.6, y=1.2, z=0.9))),
        uirevision="brain", legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
        annotations=[dict(text=f"{n_dropped} unit(s) without CCF coordinates not shown", x=0, y=0, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=10, color="gray"), xanchor="left", yanchor="bottom")] if n_dropped else [],
    )
    return fig


def unit_click_to_key(click_data) -> tuple[str, int] | None:
    """Parse a plotly click event (dict with 'points') into (session_id, unit_index)."""
    try:
        pt = click_data["points"][0]
        cd = pt.get("customdata")
        if cd is None:
            return None
        return str(cd[0]), int(cd[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
