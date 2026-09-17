#!/usr/bin/env python3
"""Standalone demo: robust context-shift units on the 3D brain, written to HTML (and PNG if kaleido exists)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from brain3d import brain_figure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t = pd.read_parquet(config.PRECOMPUTED_DIR / "context_baseline_shifts.parquet")
    rob = t[t["is_robust"]].copy()
    best = rob.assign(alr=rob["log2_ratio"].abs()).sort_values(["p_perm", "alr"], ascending=[True, False]).iloc[0]
    fig = brain_figure(rob, color_col="log2_ratio", color_range=(-3, 3), region_meshes=["MOs", "CP", "AUDp"],
                       selected_unit=(best["session_id"], int(best["unit_index"])),
                       title=f"Robust context baseline shifts (n={len(rob)}), colour = log2(aud/vis)",
                       colorbar_title="log2 aud/vis")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(a.out, include_plotlyjs="cdn")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB)")
    try:
        import kaleido  # noqa: F401
        fig.write_image(a.out.with_suffix(".png"), width=1200, height=800)
        print("wrote", a.out.with_suffix(".png"))
    except Exception as exc:  # noqa: BLE001
        print(f"no PNG (kaleido unavailable: {type(exc).__name__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
