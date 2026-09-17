#!/usr/bin/env python3
"""Screen units for context-dependent baseline (pre-stimulus) firing-rate shifts.

Design
------
Per-trial baseline rates are strongly autocorrelated within a block and drift slowly
across a session, so trial-level tests (Mann-Whitney) and naive chunk permutations are
anticonservative (on session 759434 a pure null still flagged ~38% of units with a
10-trial chunk permutation on raw rates). The primary test here is a *stratified chunk
permutation test on detrended rates*:

1. per-trial baseline rate in [START, END) s before stimulus onset, for steady-state
   trials (first N trials of every block excluded);
2. each unit's rate vector is detrended with a centred moving average whose window is
   `detrend_window_blocks` blocks of selected trials (~160 trials, truncated at edges);
3. trials are grouped into contiguous `chunk_size`-trial chunks within blocks and
   chunk context labels are permuted (preserving aud/vis chunk counts);
4. p_perm from the null distribution of mean_aud - mean_vis; BH across all units.

With defaults (window 2 blocks, chunk 20) a pure null (two same-context blocks labelled
against each other) yields ~4% of units at p<0.05 while the real labelling keeps ~45%.

Flags
-----
is_significant : q_bh < alpha                                    (liberal)
is_robust      : is_significant & block_consistency >= min_consistency
                 & max(mean_aud, mean_vis) >= min_rate          (DEFAULT for tab-1 dots)

All descriptive columns (mean rates, log2_ratio, modulation_index, block means,
block_consistency, full_block_separation, p_mwu) are computed on RAW rates.

Outputs (in --out-dir):
    context_baseline_shifts.parquet
    context_baseline_shifts_by_region.parquet
    context_baseline_shifts_by_region_session.parquet
    context_baseline_shifts_summary.json
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SCRIPT_VERSION = "1.0"
UNIT_META = ["session_id", "unit_index", "unit_id", "structure", "location", "ccf_ap", "ccf_dv",
             "ccf_ml", "task_firing_rate", "is_not_drift", "presence_ratio"]


# ----------------------------------------------------------------------------- data
def load_session(pdir: Path, sid: str):
    d = pdir / sid
    return (pd.read_parquet(d / "units.parquet"), pd.read_parquet(d / "trials.parquet"),
            pd.read_parquet(d / "aligned_spikes.parquet"))


def select_trials(trials: pd.DataFrame, exclude_first_n: int) -> pd.DataFrame:
    m = (~trials["is_instruction"]) & (~trials["is_opto"]) & (trials["trial_index_in_block"] >= exclude_first_n)
    return trials[m].sort_values("trial_index").reset_index(drop=True)


def per_trial_baseline_rates(aligned: pd.DataFrame, trials_sel: pd.DataFrame, units: pd.DataFrame,
                             window: tuple[float, float]) -> pd.DataFrame:
    """Long table [unit_index, trial_index, rate] on the full unit x selected-trial grid."""
    start, end = window
    a = aligned[(aligned["t_rel"] >= start) & (aligned["t_rel"] < end)]
    counts = a.groupby(["unit_index", "trial_index"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=units["unit_index"].to_numpy(), columns=trials_sel["trial_index"].to_numpy(), fill_value=0)
    R = counts.to_numpy(dtype=float) / (end - start)
    long = pd.DataFrame({
        "unit_index": np.repeat(units["unit_index"].to_numpy(), R.shape[1]),
        "trial_index": np.tile(trials_sel["trial_index"].to_numpy(), R.shape[0]),
        "rate": R.ravel(),
    })
    return long


def rate_matrix(rates_long: pd.DataFrame, units: pd.DataFrame, trials_sel: pd.DataFrame) -> np.ndarray:
    return (rates_long.pivot(index="unit_index", columns="trial_index", values="rate")
            .reindex(index=units["unit_index"].to_numpy(), columns=trials_sel["trial_index"].to_numpy())
            .to_numpy(dtype=float))


# ----------------------------------------------------------------------------- statistics
def detrend_rates(R: np.ndarray, block_of_trial: np.ndarray, window_blocks: float) -> np.ndarray:
    """Subtract, per unit, a centred moving average of width round(window_blocks * mean
    selected trials per block), truncated at the edges. Identity when window_blocks == 0."""
    if window_blocks == 0:
        return R.copy()
    n_t = R.shape[1]
    mean_per_block = n_t / len(np.unique(block_of_trial))
    win = int(round(window_blocks * mean_per_block))
    out = np.empty_like(R)
    cs = np.concatenate([np.zeros((R.shape[0], 1)), np.cumsum(R, axis=1)], axis=1)
    for j in range(n_t):
        lo, hi = max(0, j - win // 2), min(n_t, j + win // 2)
        out[:, j] = R[:, j] - (cs[:, hi] - cs[:, lo]) / (hi - lo)
    return out


def make_chunks(block_of_trial: np.ndarray, chunk_size: int) -> np.ndarray:
    """Contiguous chunk ids within blocks; a final chunk shorter than chunk_size//2 merges into the previous."""
    cid = np.empty(len(block_of_trial), dtype=int)
    c = 0
    for b in np.unique(block_of_trial):
        idx = np.where(block_of_trial == b)[0]
        n = len(idx)
        bounds = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]
        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < chunk_size // 2:
            bounds[-2] = (bounds[-2][0], bounds[-1][1])
            bounds.pop()
        for s, e in bounds:
            cid[idx[s:e]] = c
            c += 1
    return cid


def contrast_weights(ctx: np.ndarray) -> np.ndarray:
    """+1/n_aud for aud trials, -1/n_vis for vis trials, so R @ w = mean_aud - mean_vis."""
    n1 = ctx.sum()
    n0 = len(ctx) - n1
    return np.where(ctx == 1, 1.0 / n1, -1.0 / n0)


def chunk_permutation_test(R_d: np.ndarray, ctx: np.ndarray, block_of_trial: np.ndarray, chunk_size: int,
                           n_perm: int, rng: np.random.Generator):
    """Return (obs_diff, p_perm, n_chunks_aud, n_chunks_vis). ctx: 1 = aud, 0 = vis."""
    cid = make_chunks(block_of_trial, chunk_size)
    n_chunks = cid.max() + 1
    chunk_ctx = np.array([ctx[cid == c][0] for c in range(n_chunks)])
    obs = R_d @ contrast_weights(ctx)
    W = np.empty((R_d.shape[1], n_perm))
    for k in range(n_perm):
        W[:, k] = contrast_weights(rng.permutation(chunk_ctx)[cid])
    P = R_d @ W
    p = (1 + (np.abs(P) >= np.abs(obs)[:, None]).sum(axis=1)) / (n_perm + 1)
    return obs, p, int(chunk_ctx.sum()), int(n_chunks - chunk_ctx.sum())


def unit_stats(R: np.ndarray, units: pd.DataFrame, trials_sel: pd.DataFrame, chunk_size: int,
               detrend_window_blocks: float, n_perm: int, rng: np.random.Generator,
               ctx_override: np.ndarray | None = None) -> pd.DataFrame:
    """Per-unit table for one session (without q_bh / flags, which are computed across sessions)."""
    block = trials_sel["block_index"].to_numpy()
    ctx = (trials_sel["rewarded_modality"].to_numpy() == "aud").astype(int) if ctx_override is None else ctx_override
    aud, vis = ctx == 1, ctx == 0

    mean_aud, mean_vis = R[:, aud].mean(axis=1), R[:, vis].mean(axis=1)
    med_aud, med_vis = np.median(R[:, aud], axis=1), np.median(R[:, vis], axis=1)
    obs_diff = mean_aud - mean_vis
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio = np.log2((mean_aud + 0.1) / (mean_vis + 0.1))
        mod_idx = np.where(mean_aud + mean_vis > 0, (mean_aud - mean_vis) / (mean_aud + mean_vis), np.nan)
    direction = np.where(obs_diff > 0, "aud_higher", np.where(obs_diff < 0, "vis_higher", "none"))

    # primary test on detrended rates
    R_d = detrend_rates(R, block, detrend_window_blocks)
    obs_d, p_perm, n_ch_aud, n_ch_vis = chunk_permutation_test(R_d, ctx, block, chunk_size, n_perm, rng)

    # descriptive Mann-Whitney on raw rates
    p_mwu = np.ones(len(R))
    for i in range(len(R)):
        a, v = R[i, aud], R[i, vis]
        if len(a) == 0 or len(v) == 0 or np.ptp(np.concatenate([a, v])) == 0:
            continue
        p_mwu[i] = stats.mannwhitneyu(a, v, alternative="two-sided").pvalue

    # block-level metrics on raw rates
    blocks = np.unique(block)
    block_means = np.stack([R[:, block == b].mean(axis=1) for b in blocks], axis=1)  # units x blocks
    block_ctx = np.array([ctx[block == b][0] for b in blocks])
    sign_obs = np.sign(obs_diff)
    matches = []
    for k in range(len(blocks) - 1):
        b0, b1 = k, k + 1
        if block_ctx[b0] == block_ctx[b1]:
            continue  # non-alternating pair carries no context contrast
        aud_b, vis_b = (b0, b1) if block_ctx[b0] == 1 else (b1, b0)
        matches.append(np.sign(block_means[:, aud_b] - block_means[:, vis_b]) == sign_obs)
    if matches:
        consistency = np.mean(np.stack(matches, axis=1), axis=1)
    else:
        consistency = np.full(len(R), np.nan)
    consistency = np.where(sign_obs == 0, np.nan, consistency)
    aud_cols, vis_cols = block_means[:, block_ctx == 1], block_means[:, block_ctx == 0]
    full_sep = (aud_cols.min(axis=1) > vis_cols.max(axis=1)) | (vis_cols.min(axis=1) > aud_cols.max(axis=1))

    out = units[UNIT_META].copy().reset_index(drop=True)
    out["n_trials_aud"], out["n_trials_vis"] = int(aud.sum()), int(vis.sum())
    out["mean_rate_aud"], out["mean_rate_vis"] = mean_aud, mean_vis
    out["median_rate_aud"], out["median_rate_vis"] = med_aud, med_vis
    out["obs_diff"], out["log2_ratio"], out["modulation_index"] = obs_diff, log2_ratio, mod_idx
    out["shift_direction"] = direction
    out["obs_diff_detrended"], out["p_perm"] = obs_d, p_perm
    out["n_chunks_aud"], out["n_chunks_vis"] = n_ch_aud, n_ch_vis
    out["p_mwu"] = p_mwu
    for k, b in enumerate(blocks):
        out[f"block_mean_{int(b)}"] = block_means[:, k]
    out["block_consistency"] = consistency
    out["full_block_separation"] = full_sep
    return out


def add_flags(table: pd.DataFrame, alpha: float, min_consistency: float, min_rate: float) -> pd.DataFrame:
    table = table.copy()
    table["q_bh"] = stats.false_discovery_control(table["p_perm"].to_numpy(), method="bh")
    table["is_significant"] = table["q_bh"] < alpha
    max_rate = np.maximum(table["mean_rate_aud"], table["mean_rate_vis"])
    table["is_robust"] = (table["is_significant"] & (table["block_consistency"] >= min_consistency)
                          & (max_rate >= min_rate)).fillna(False).astype(bool)
    return table


def region_summary(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        rob = g[g["is_robust"]]
        return pd.Series({
            "n_units": len(g),
            "n_significant": int(g["is_significant"].sum()),
            "n_robust": int(g["is_robust"].sum()),
            "frac_significant": g["is_significant"].mean(),
            "frac_robust": g["is_robust"].mean(),
            "n_aud_higher_robust": int((rob["shift_direction"] == "aud_higher").sum()),
            "n_vis_higher_robust": int((rob["shift_direction"] == "vis_higher").sum()),
            "median_abs_log2_ratio_robust": rob["log2_ratio"].abs().median() if len(rob) else np.nan,
            "n_sessions": g["session_id"].nunique() if "session_id" in g.columns else 1,
        })
    out = t.groupby(by, sort=True).apply(agg, include_groups=False).reset_index()
    if by == ["structure"]:
        out = out.sort_values(["frac_robust", "n_units"], ascending=[False, False])
    else:
        out = out.drop(columns="n_sessions")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precomputed-dir", type=Path, default=config.PRECOMPUTED_DIR)
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--baseline-window", nargs=2, type=float, default=(-1.5, -0.06), metavar=("START", "END"))
    ap.add_argument("--exclude-first-n", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--detrend-window-blocks", type=float, default=2.0)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--min-rate", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-consistency", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=config.PRECOMPUTED_DIR)
    a = ap.parse_args()
    window = (float(a.baseline_window[0]), float(a.baseline_window[1]))

    sids = a.sessions or sorted(p.parent.name for p in a.precomputed_dir.glob("*/aligned_spikes.parquet"))
    t0 = time.time()
    tables = []
    for sid in sids:
        ts = time.time()
        rng = np.random.default_rng(a.seed)
        units, trials, aligned = load_session(a.precomputed_dir, sid)
        sel = select_trials(trials, a.exclude_first_n)
        rates = per_trial_baseline_rates(aligned, sel, units, window)
        R = rate_matrix(rates, units, sel)
        t = unit_stats(R, units, sel, a.chunk_size, a.detrend_window_blocks, a.n_perm, rng)
        tables.append(t)
        print(f"[ok] {sid}: {len(t)} units, {len(sel)} selected trials, {time.time() - ts:.1f}s")

    table = pd.concat(tables, ignore_index=True)
    table = add_flags(table, a.alpha, a.min_consistency, a.min_rate)
    table = table.sort_values(["session_id", "unit_index"]).reset_index(drop=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(a.out_dir / "context_baseline_shifts.parquet", index=False)
    by_region = region_summary(table, ["structure"])
    by_region.to_parquet(a.out_dir / "context_baseline_shifts_by_region.parquet", index=False)
    by_region_session = region_summary(table, ["session_id", "structure"])
    by_region_session.to_parquet(a.out_dir / "context_baseline_shifts_by_region_session.parquet", index=False)

    n = len(table)
    n_sig, n_rob = int(table["is_significant"].sum()), int(table["is_robust"].sum())
    per_session = table.groupby("session_id").agg(n_units=("unit_index", "size"), n_significant=("is_significant", "sum"),
                                                  n_robust=("is_robust", "sum")).astype(int)
    top = by_region[by_region["n_units"] >= 20].head(15)
    print(f"\nTotal units {n}: significant (q<{a.alpha}) {n_sig} ({n_sig / n:.1%}), robust {n_rob} ({n_rob / n:.1%}), "
          f"full block separation {table['full_block_separation'].mean():.1%}")
    print("\nPer session:\n" + per_session.to_string())
    print("\nTop structures by frac_robust (n>=20):\n" + top[["structure", "n_units", "n_robust", "frac_robust",
          "n_aud_higher_robust", "n_vis_higher_robust", "n_sessions"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    runtime = time.time() - t0
    print(f"\nruntime {runtime:.1f}s")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "params": {"baseline_window": list(window), "exclude_first_n": a.exclude_first_n, "chunk_size": a.chunk_size,
                   "detrend_window_blocks": a.detrend_window_blocks, "n_perm": a.n_perm, "min_rate": a.min_rate,
                   "alpha": a.alpha, "min_consistency": a.min_consistency, "seed": a.seed},
        "flag_semantics": {"is_robust": "DEFAULT flag for tab-1 dots: q_bh<alpha on the detrended chunk-permutation test, "
                                        "block_consistency>=min_consistency, max context mean rate>=min_rate",
                           "is_significant": "liberal flag: q_bh<alpha only"},
        "n_units": n, "n_significant": n_sig, "n_robust": n_rob,
        "frac_full_block_separation": float(table["full_block_separation"].mean()),
        "n_robust_aud_higher": int((table["is_robust"] & (table["shift_direction"] == "aud_higher")).sum()),
        "n_robust_vis_higher": int((table["is_robust"] & (table["shift_direction"] == "vis_higher")).sum()),
        "per_session": per_session.reset_index().to_dict(orient="records"),
        "top_structures": top.to_dict(orient="records"),
        "runtime_s": round(runtime, 1),
    }
    (a.out_dir / "context_baseline_shifts_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
