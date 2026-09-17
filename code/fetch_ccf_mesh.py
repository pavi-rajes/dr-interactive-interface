#!/usr/bin/env python3
"""Fetch Allen CCF meshes (via brainglobe-atlasapi) into PRECOMPUTED_DIR/ccf_meshes.

Saves root + every structure present in units_all.parquet (except 'out of brain') as
<acronym>.npz with `vertices` (float32, micrometres, atlas axis order ap/dv/ml) and
`faces` (int32 triangles), plus index.json. If `fast_simplification` is importable a
decimated root mesh is also written as root_decimated.npz.

Usage: python code/fetch_ccf_mesh.py [--atlas allen_mouse_25um] [--out-dir DIR]
                                     [--structures ACR ...] [--decimate-target 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def mesh_arrays(atlas, acronym: str) -> tuple[np.ndarray, np.ndarray]:
    m = atlas.mesh_from_structure(acronym)
    verts = np.asarray(m.points, dtype=np.float32)
    tri = [c for c in m.cells if c.type == "triangle"]
    faces = np.asarray(tri[0].data, dtype=np.int32)
    return verts, faces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", default="allen_mouse_25um")
    ap.add_argument("--out-dir", type=Path, default=config.PRECOMPUTED_DIR / "ccf_meshes")
    ap.add_argument("--structures", nargs="+", default=None)
    ap.add_argument("--decimate-target", type=float, default=0.25, help="fraction of faces to keep for root_decimated")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402
    t0 = time.time()
    atlas = BrainGlobeAtlas(a.atlas, check_latest=False)
    if a.structures is None:
        units = pd.read_parquet(config.PRECOMPUTED_DIR / "units_all.parquet")
        a.structures = sorted(s for s in units["structure"].dropna().unique() if s != "out of brain")

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "atlas": a.atlas, "orientation": atlas.orientation, "resolution_um": list(atlas.resolution),
        "axis_order": ["ap", "dv", "ml"], "structures": {}, "missing": [], "decimated": False,
    }
    for acr in ["root"] + list(a.structures):
        try:
            verts, faces = mesh_arrays(atlas, acr)
        except Exception as exc:  # noqa: BLE001
            index["missing"].append({"acronym": acr, "error": f"{type(exc).__name__}: {exc}"})
            continue
        np.savez_compressed(a.out_dir / f"{acr}.npz", vertices=verts, faces=faces)
        st = atlas.structures[acr]
        index["structures"][acr] = {
            "acronym": acr, "id": int(st["id"]), "name": st["name"], "rgb": [int(c) for c in st["rgb_triplet"]],
            "n_vertices": int(len(verts)), "n_faces": int(len(faces)), "file": f"{acr}.npz",
            "bounds_min_um": verts.min(0).astype(float).round(2).tolist(), "bounds_max_um": verts.max(0).astype(float).round(2).tolist(),
        }
    try:
        import fast_simplification  # noqa: E402
        verts, faces = mesh_arrays(atlas, "root")
        dv, df = fast_simplification.simplify(verts.astype(np.float64), faces, target_reduction=1 - a.decimate_target)
        np.savez_compressed(a.out_dir / "root_decimated.npz", vertices=np.asarray(dv, dtype=np.float32), faces=np.asarray(df, dtype=np.int32))
        index["decimated"] = {"file": "root_decimated.npz", "n_vertices": int(len(dv)), "n_faces": int(len(df)), "target_fraction": a.decimate_target}
    except Exception as exc:  # noqa: BLE001
        index["decimated"] = False
        index["decimation_error"] = f"{type(exc).__name__}: {exc}"
    (a.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(p.stat().st_size for p in a.out_dir.glob("*.npz"))
    print(f"wrote {len(index['structures'])} meshes ({len(index['missing'])} missing), decimated={bool(index['decimated'])}, "
          f"{total / 1e6:.1f} MB, {time.time() - t0:.1f}s -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
