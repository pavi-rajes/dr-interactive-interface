#!/usr/bin/env python3
"""Verify CCF mesh export and unit/mesh alignment. Exits non-zero on failure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from brain3d import load_mesh  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def main() -> int:
    from brainglobe_atlasapi import BrainGlobeAtlas
    mesh_dir = config.PRECOMPUTED_DIR / "ccf_meshes"
    index = json.loads((mesh_dir / "index.json").read_text())
    atlas = BrainGlobeAtlas(index["atlas"], check_latest=False)
    units = pd.read_parquet(config.PRECOMPUTED_DIR / "units_all.parquet").dropna(subset=["ccf_ap", "ccf_dv", "ccf_ml"])
    expected = sorted(s for s in units["structure"].unique() if s != "out of brain")
    check("all structure meshes written", all((mesh_dir / f"{s}.npz").exists() for s in expected) and (mesh_dir / "root.npz").exists(),
          f"{len(expected)} structures + root; missing in index: {index['missing']}")
    rv, _ = load_mesh("root", mesh_dir, prefer_decimated=False)
    av = np.asarray(atlas.mesh_from_structure("root").points)
    check("root bounds match atlas within 1 um", np.allclose(rv.min(0), av.min(0), atol=1) and np.allclose(rv.max(0), av.max(0), atol=1),
          f"npz min {rv.min(0).round(1)} max {rv.max(0).round(1)}")

    for acr in ["MOs", "CP", "AUDp", "SSp", "ACAd", "ORBvl", "VISp"]:
        m = units["structure"].str.startswith("SSp") if acr == "SSp" else units["structure"] == acr
        u = units[m]
        v, _ = load_mesh(acr, mesh_dir)
        lo, hi = v.min(0) - 100, v.max(0) + 100
        pts = u[["ccf_ap", "ccf_dv", "ccf_ml"]].to_numpy(dtype=float)
        in_box = np.all((pts >= lo) & (pts <= hi), axis=1).mean()
        desc = set(atlas.get_structure_descendants(acr))
        inside = 0
        for p in pts:
            lab = atlas.structure_from_coords(tuple(p), microns=True, as_acronym=True)
            if lab == acr or lab in desc:
                inside += 1
            elif lab in atlas.structures and acr in atlas.get_structure_ancestors(lab):
                inside += 1
        frac_in = inside / len(pts)
        d, _ = cKDTree(v).query(pts)
        check(f"{acr}: bbox+100um fraction >= 0.95", in_box >= 0.95, f"{in_box:.3f} (n={len(u)})")
        check(f"{acr}: annotation containment >= 0.95", frac_in >= 0.95, f"{frac_in:.3f}; median nearest-surface-vertex {np.median(d):.0f} um (info)")

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
