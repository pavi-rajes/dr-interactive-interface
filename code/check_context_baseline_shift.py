#!/usr/bin/env python3
"""Sanity checks and null calibration for analyze_context_baseline_shift.py.

Usage: python code/check_context_baseline_shift.py [--session 759434_2025-02-04] [--n-perm 500]
Exits non-zero on any FAIL.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from analyze_context_baseline_shift import (chunk_permutation_test, detrend_rates, load_session,  # noqa: E402
                                            per_trial_baseline_rates, rate_matrix, select_trials, unit_stats)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="759434_2025-02-04")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--precomputed-dir", type=Path, default=config.PRECOMPUTED_DIR)
    a = ap.parse_args()
    pdir = a.precomputed_dir
    summ = json.loads((pdir / "context_baseline_shifts_summary.json").read_text())
    P = summ["params"]
    window = tuple(P["baseline_window"])
    table = pd.read_parquet(pdir / "context_baseline_shifts.parquet")
    units, trials, aligned = load_session(pdir, a.session)
    sel = select_trials(trials, P["exclude_first_n"])
    R = rate_matrix(per_trial_baseline_rates(aligned, sel, units, window), units, sel)
    block = sel["block_index"].to_numpy()
    true_ctx = (sel["rewarded_modality"].to_numpy() == "aud").astype(int)
    tsess = table[table["session_id"] == a.session].set_index("unit_index")

    # (a) independent recomputation of context means for 3 robust units
    rng = np.random.default_rng(0)
    rob = tsess[tsess["is_robust"]].index.to_numpy()
    picks = rng.choice(rob, size=3, replace=False)
    ok, details = True, []
    sel_idx = set(sel["trial_index"].tolist())
    aud_trials = set(sel.loc[sel["rewarded_modality"] == "aud", "trial_index"].tolist())
    for ui in picks:
        spk = aligned[(aligned["unit_index"] == ui) & (aligned["t_rel"] >= window[0]) & (aligned["t_rel"] < window[1])]
        rates = {}
        for ctx_name, tset in (("aud", aud_trials), ("vis", sel_idx - aud_trials)):
            cnt = sum(int((spk["trial_index"] == ti).sum()) for ti in tset)
            rates[ctx_name] = cnt / (len(tset) * (window[1] - window[0]))
        ma, mv = tsess.loc[ui, "mean_rate_aud"], tsess.loc[ui, "mean_rate_vis"]
        if abs(rates["aud"] - ma) > 1e-6 or abs(rates["vis"] - mv) > 1e-6:
            ok = False
        details.append(f"u{int(ui)} aud {rates['aud']:.3f}/{ma:.3f} vis {rates['vis']:.3f}/{mv:.3f}")
    check("(a) independent recount of context means for 3 robust units", ok, "; ".join(details))

    # (b) calibration
    R_d = detrend_rates(R, block, P["detrend_window_blocks"])
    n_perm, cs = a.n_perm, P["chunk_size"]

    # (i) pure null: same-context block pairs
    fps = []
    for b1, b2 in [(0, 2), (2, 4), (0, 4), (1, 3), (3, 5), (1, 5)]:
        m = np.isin(block, [b1, b2])
        ctx = (block[m] == b1).astype(int)
        _, p, _, _ = chunk_permutation_test(R_d[:, m], ctx, block[m], cs, n_perm, np.random.default_rng(1))
        fps.append((p < 0.05).mean())
    check("(b-i) pure null (same-context block pairs) frac p<0.05 mean<=0.10 and max<=0.15",
          np.mean(fps) <= 0.10 and np.max(fps) <= 0.15, f"mean {np.mean(fps):.3f} max {np.max(fps):.3f} per pair {np.round(fps, 3).tolist()}")

    # (ii) rank of real labelling among 19 block labellings
    true_aud = tuple(sorted(np.unique(block[true_ctx == 1]).tolist()))
    def frac_sig(ctx, Rx, rng):
        _, p, _, _ = chunk_permutation_test(Rx, ctx, block, cs, n_perm, rng)
        return (p < 0.05).mean()
    rng = np.random.default_rng(0)
    real = frac_sig(true_ctx, R_d, rng)
    shuf = []
    for comb in itertools.combinations(range(6), 3):
        if comb == true_aud or tuple(sorted(set(range(6)) - set(comb))) == true_aud:
            continue
        shuf.append(frac_sig(np.isin(block, comb).astype(int), R_d, rng))
    shuf = np.array(shuf)
    rank = 1 + int((shuf > real).sum())
    check("(b-ii) real labelling ranks 1st of 19 block labellings (detrended chunk test)", rank == 1,
          f"real {real:.3f} | shuffles mean {np.mean(shuf):.3f} max {np.max(shuf):.3f} (n={len(shuf)}; shuffles agree with truth on 2 or 4 of 6 blocks, so only partially null) | rank {rank}/19")

    # (iii) documentation: same comparison for p_mwu and undetrended chunk test
    def frac_mwu(ctx):
        aud, vis = ctx == 1, ctx == 0
        ps = np.array([stats.mannwhitneyu(R[i, aud], R[i, vis]).pvalue if np.ptp(R[i]) > 0 else 1.0 for i in range(len(R))])
        return (ps < 0.05).mean()
    mwu_real = frac_mwu(true_ctx)
    mwu_shuf = np.array([frac_mwu(np.isin(block, comb).astype(int)) for comb in itertools.combinations(range(6), 3)
                         if comb != true_aud and tuple(sorted(set(range(6)) - set(comb))) != true_aud])
    rng = np.random.default_rng(0)
    raw_real = frac_sig(true_ctx, R, rng)
    raw_shuf = np.array([frac_sig(np.isin(block, comb).astype(int), R, rng) for comb in itertools.combinations(range(6), 3)
                         if comb != true_aud and tuple(sorted(set(range(6)) - set(comb))) != true_aud])
    print(f"[info] (b-iii) trial-level MWU: real {mwu_real:.3f} | shuffles mean {mwu_shuf.mean():.3f} max {mwu_shuf.max():.3f}  (anticonservative, descriptive only)")
    print(f"[info] (b-iii) undetrended chunk test: real {raw_real:.3f} | shuffles mean {raw_shuf.mean():.3f} max {raw_shuf.max():.3f}")

    # (c) flag consistency
    check("(c1) q_bh >= p_perm for all rows", bool((table["q_bh"] >= table["p_perm"] - 1e-12).all()))
    check("(c2) is_robust implies is_significant", bool((~table["is_robust"] | table["is_significant"]).all()))
    check("(c3) p_perm >= 1/(n_perm+1)", bool((table["p_perm"] >= 1 / (P["n_perm"] + 1) - 1e-12).all()), f"min p_perm {table['p_perm'].min():.5f}")

    # (d) detrend sanity
    const = np.full((3, R.shape[1]), 2.5)
    check("(d1) constant-rate input detrends to all zeros (edges included)", np.abs(detrend_rates(const, block, P["detrend_window_blocks"])).max() < 1e-9)
    check("(d2) detrend_rates(..., 0) returns R unchanged", np.array_equal(detrend_rates(R, block, 0), R))
    ui = int(picks[0]); row = int(np.where(units["unit_index"].to_numpy() == ui)[0][0])
    win = int(round(P["detrend_window_blocks"] * R.shape[1] / len(np.unique(block))))
    interior = slice(win // 2, R.shape[1] - win // 2)
    print(f"[info] (d3) unit {ui}: interior mean raw {R[row, interior].mean():.4f} Hz vs detrended {R_d[row, interior].mean():.4f} Hz (window {win} trials)")

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed for {a.session}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
