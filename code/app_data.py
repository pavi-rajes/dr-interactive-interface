"""Data access layer for the Shiny app: reads only precomputed parquet/JSON, never NWB."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config

P = config.PRECOMPUTED_DIR


@lru_cache(maxsize=1)
def sessions() -> list[str]:
    return sorted(p.parent.name for p in Path(P).glob("*/aligned_spikes.parquet"))


@lru_cache(maxsize=8)
def load_units(sid: str) -> pd.DataFrame:
    return pd.read_parquet(P / sid / "units.parquet")


@lru_cache(maxsize=8)
def load_trials(sid: str) -> pd.DataFrame:
    return pd.read_parquet(P / sid / "trials.parquet").sort_values("trial_index").reset_index(drop=True)


@lru_cache(maxsize=8)
def load_aligned(sid: str) -> pd.DataFrame:
    df = pd.read_parquet(P / sid / "aligned_spikes.parquet")
    if not df["unit_index"].is_monotonic_increasing:
        df = df.sort_values(["unit_index", "trial_index"], kind="stable").reset_index(drop=True)
    return df


def unit_spikes(sid: str, unit_index: int) -> pd.DataFrame:
    """[trial_index, t_rel] for one unit (fast slice on the unit-sorted table)."""
    al = load_aligned(sid)
    col = al["unit_index"].to_numpy()
    lo, hi = np.searchsorted(col, unit_index, side="left"), np.searchsorted(col, unit_index, side="right")
    return al.iloc[lo:hi][["trial_index", "t_rel"]]


@lru_cache(maxsize=1)
def context_table() -> pd.DataFrame:
    return pd.read_parquet(P / "context_baseline_shifts.parquet")


@lru_cache(maxsize=1)
def context_regions() -> pd.DataFrame:
    return pd.read_parquet(P / "context_baseline_shifts_by_region.parquet")


@lru_cache(maxsize=1)
def rule_table() -> pd.DataFrame:
    return pd.read_parquet(P / "rule_change_units.parquet")


@lru_cache(maxsize=1)
def rule_regions() -> pd.DataFrame:
    return pd.read_parquet(P / "rule_change_by_region.parquet")


@lru_cache(maxsize=1)
def rule_profiles() -> pd.DataFrame:
    return pd.read_parquet(P / "rule_change_profiles.parquet")


@lru_cache(maxsize=1)
def region_profiles() -> pd.DataFrame:
    return pd.read_parquet(P / "rule_change_profiles_by_region.parquet")


@lru_cache(maxsize=1)
def summaries() -> dict:
    return {"context": json.loads((P / "context_baseline_shifts_summary.json").read_text()),
            "rule": json.loads((P / "rule_change_summary.json").read_text())}


@lru_cache(maxsize=1)
def recorded_units_by_structure() -> dict:
    """{structure: {n_units, n_units_qc}} summed over sessions from sessions_manifest.json (all recorded units, incl. QC-fail)."""
    m = json.loads((P / "sessions_manifest.json").read_text())
    out: dict = {}
    for s in m["sessions"].values():
        for acr, v in s.get("units", {}).get("structures", {}).items():
            d = out.setdefault(acr, {"n_units": 0, "n_units_qc": 0})
            d["n_units"] += int(v["n_units"]); d["n_units_qc"] += int(v["n_units_qc"])
    return out


def structures_for(table: pd.DataFrame, sid: str, flag_col: str | None, only_flagged: bool) -> pd.DataFrame:
    """[structure, n_units, n_flagged] for one session, optionally only structures with flagged units.
    flag_col None means every unit counts as flagged."""
    t = table[table["session_id"] == sid]
    if flag_col is None:
        t = t.assign(_all=True); flag_col = "_all"
    g = t.groupby("structure").agg(n_units=("unit_index", "size"), n_flagged=(flag_col, "sum")).reset_index()
    g["n_flagged"] = g["n_flagged"].astype(int)
    if only_flagged:
        g = g[g["n_flagged"] > 0]
    return g.sort_values(["n_flagged", "n_units"], ascending=[False, False]).reset_index(drop=True)


def unit_profile(sid: str, unit_index: int, window: str) -> pd.DataFrame:
    """Block-start and block-end aligned mean profiles for one unit (all contexts) in one window."""
    p = rule_profiles()
    return p[(p["session_id"] == sid) & (p["unit_index"] == unit_index) & (p["window"] == window)]


def structure_profile(structure: str, window: str) -> pd.DataFrame:
    p = region_profiles()
    return p[(p["structure"] == structure) & (p["window"] == window)]
