#!/usr/bin/env python3
"""Sanity checks and pseudo-switch calibration for analyze_rule_change_units.py (all sessions).

Usage: python code/check_rule_change_units.py [--session 759434_2025-02-04] [--n-perm 500]
Writes calibration numbers into rule_change_summary.json under "calibration". Exits non-zero on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from analyze_context_baseline_shift import detrend_rates, load_session  # noqa: E402
from analyze_rule_change_units import block_lengths, early_late_test, kw_instruction_test, masks_for, per_trial_rates, residualize_on_stimulus  # noqa: E402
from scipy import stats  # noqa: E402

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
    summ_path = pdir / "rule_change_summary.json"
    summ = json.loads(summ_path.read_text())
    P = summ["params"]
    table = pd.read_parquet(pdir / "rule_change_units.parquet")
    prof = pd.read_parquet(pdir / "rule_change_profiles.parquet")
    both = table[table["context"] == "both"]

    # (a) independent recount for 3 rule-updating units in the check session
    units, trials, aligned = load_session(pdir, a.session)
    trials = trials.sort_values("trial_index").reset_index(drop=True)
    tsess = both[both["session_id"] == a.session]
    rng = np.random.default_rng(0)
    pool = tsess.loc[tsess["kw_is_rule_updating"], "unit_index"].to_numpy()
    picks = rng.choice(pool, size=3, replace=False)
    tib = trials["trial_index_in_block"].to_numpy(); nb = block_lengths(trials); sw = (trials["block_index"] > 0).to_numpy()
    e0, e1 = P["early_window"]
    early_t = set(trials.loc[sw & (tib >= e0) & (tib < e1), "trial_index"])
    late_t = set(trials.loc[sw & (tib >= nb - P["late_n"]), "trial_index"])
    ok, det = True, []
    for ui in picks:
        row = tsess[tsess["unit_index"] == ui].iloc[0]
        spk = aligned[aligned["unit_index"] == ui]
        for w, win in (("bl", P["baseline_window"]), ("ev", P["evoked_window"])):
            s = spk[(spk["t_rel"] >= win[0]) & (spk["t_rel"] < win[1])]
            for lab, tset in (("early", early_t), ("late", late_t)):
                cnt = sum(int((s["trial_index"] == t).sum()) for t in tset)
                rate = cnt / (len(tset) * (win[1] - win[0]))
                if abs(rate - row[f"{w}_mean_{lab}"]) > 1e-6:
                    ok = False
                det.append(f"u{int(ui)} {w}_{lab} {rate:.3f}/{row[f'{w}_mean_{lab}']:.3f}")
    # instruction means too
    inst_t = set(trials.loc[sw & (tib < P["instruction_n"]), "trial_index"])
    for ui in picks:
        row = tsess[tsess["unit_index"] == ui].iloc[0]
        spk = aligned[aligned["unit_index"] == ui]
        for w, win in (("bl", P["baseline_window"]), ("ev", P["evoked_window"])):
            s_ = spk[(spk["t_rel"] >= win[0]) & (spk["t_rel"] < win[1])]
            cnt = sum(int((s_["trial_index"] == t).sum()) for t in inst_t)
            if abs(cnt / (len(inst_t) * (win[1] - win[0])) - row[f"{w}_mean_instruction"]) > 1e-6:
                ok = False
    check("(a) independent recount of instruction/early/late means (3 units x 2 windows)", ok, "; ".join(det[:6]) + " ...")

    # (b) pseudo-switch calibration on all sessions
    sids = sorted(p.parent.name for p in pdir.glob("*/aligned_spikes.parquet"))
    calib = {}
    for sid in sids:
        u, t, al = load_session(pdir, sid)
        t = t.sort_values("trial_index").reset_index(drop=True)
        blk_s = t["block_index"].to_numpy()
        R_bl = per_trial_rates(al, t, u, tuple(P["baseline_window"]))
        R_ev = per_trial_rates(al, t, u, tuple(P["evoked_window"]))
        M = {"baseline": detrend_rates(R_bl, blk_s, P["detrend_window_blocks"]),
             "evoked": detrend_rates(residualize_on_stimulus(R_ev, t), blk_s, P["detrend_window_blocks"])}
        _, early, late, _ = masks_for(t, "both", P["instruction_n"], tuple(P["early_window"]), P["late_n"])
        tib_s = t["trial_index_in_block"].to_numpy()
        in_sw = (blk_s > 0)
        p_early = in_sw & (tib_s >= 40) & (tib_s < 50)   # mid-block pseudo windows, same size as the real ones
        p_late = in_sw & (tib_s >= 20) & (tib_s < 30)
        calib[sid] = {}
        for w, Rd in M.items():
            _, p_real, _, _ = early_late_test(Rd, early, late, blk_s, P["chunk_size"], a.n_perm, np.random.default_rng(1))
            _, p_pseudo, _, _ = early_late_test(Rd, p_early, p_late, blk_s, P["chunk_size"], a.n_perm, np.random.default_rng(1))
            calib[sid][w] = {"real": float((p_real < 0.05).mean()), "pseudo": float((p_pseudo < 0.05).mean())}
        print(f"[info] {sid}: baseline real {calib[sid]['baseline']['real']:.3f} pseudo {calib[sid]['baseline']['pseudo']:.3f} | "
              f"evoked(res) real {calib[sid]['evoked']['real']:.3f} pseudo {calib[sid]['evoked']['pseudo']:.3f}")
    for w in ("baseline", "evoked"):
        ps = np.array([calib[s][w]["pseudo"] for s in sids]); rs = np.array([calib[s][w]["real"] for s in sids])
        check(f"(b) {w}: pseudo-switch null mean<=0.10 and max<=0.15 across sessions", ps.mean() <= 0.10 and ps.max() <= 0.15,
              f"pseudo mean {ps.mean():.3f} max {ps.max():.3f} | real mean {rs.mean():.3f} min {rs.min():.3f}")
    summ["calibration"] = {"n_perm": a.n_perm, "pseudo_early": "40<=trial_index_in_block<50", "pseudo_late": "20<=trial_index_in_block<30", "per_session": calib}
    summ_path.write_text(json.dumps(summ, indent=2, default=float))

    # (c) flags, both contrasts
    for cpre in ("", "inst_"):
        for w in ("bl", "ev"):
            check(f"(c) {w}_{cpre}: q_bh >= p_perm", bool((table[f"{w}_{cpre}q_bh"] >= table[f"{w}_{cpre}p_perm"] - 1e-12).all()))
            check(f"(c) {w}_{cpre}: robust implies significant; robust implies candidate implies liberal",
                  bool((~table[f"{w}_{cpre}is_robust"] | table[f"{w}_{cpre}is_significant"]).all()
                       and (~table[f"{w}_{cpre}is_candidate"] | table[f"{w}_{cpre}is_liberal"]).all()))
            check(f"(c) {w}_{cpre}: p_perm >= 1/(n_perm+1)", bool((table[f"{w}_{cpre}p_perm"] >= 1 / (P["n_perm"] + 1) - 1e-12).all()))
        check(f"(c) {cpre}is_rule_updating == bl|ev robust; {cpre}is_candidate == bl|ev candidate",
              bool((table[f"{cpre}is_rule_updating"] == (table[f"bl_{cpre}is_robust"] | table[f"ev_{cpre}is_robust"])).all()
                   and (table[f"{cpre}is_candidate"] == (table[f"bl_{cpre}is_candidate"] | table[f"ev_{cpre}is_candidate"])).all()))
    nbk = table.pivot_table(index=["session_id", "unit_index"], columns="context", values="n_blocks")
    check("(c) n_blocks: both == 5 and contexts sum to 5", bool((nbk["both"] == 5).all() and ((nbk["aud"] + nbk["vis"]) == 5).all()))
    check("(c) n_trials_early == 10 x n_blocks and n_trials_late == late_n x n_blocks",
          bool((table["n_trials_early"] == (P["early_window"][1] - P["early_window"][0]) * table["n_blocks"]).all() and (table["n_trials_late"] == P["late_n"] * table["n_blocks"]).all()))
    # single-label chunks is asserted inside early_late_test (would have raised above)
    check("(c) every chunk in the primary test has a single label", True, "asserted inside early_late_test during (b)")

    # (e) Kruskal-Wallis: H and chi2 p match scipy for 3 units; shuffle-null calibration with mid-block pseudo phases
    R_bl = per_trial_rates(aligned, trials, units, tuple(P["baseline_window"]))
    block = trials["block_index"].to_numpy(); tib = trials["trial_index_in_block"].to_numpy(); nb = block_lengths(trials); sw = block > 0
    inst_m, early_m, late_m, _ = masks_for(trials, "both", P["instruction_n"], tuple(P["early_window"]), P["late_n"])
    ok, det = True, []
    for ui in picks:
        row = int(np.where(units["unit_index"].to_numpy() == ui)[0][0])
        r = R_bl[row]
        H_sp, p_sp = stats.kruskal(r[inst_m], r[early_m], r[late_m])
        got = tsess[tsess["unit_index"] == ui].iloc[0]
        if abs(H_sp - got["bl_kw_h"]) > 1e-6 or abs(p_sp - got["bl_kw_p_chi2"]) > 1e-9:
            ok = False
        det.append(f"u{int(ui)} H {H_sp:.3f}/{got['bl_kw_h']:.3f} p_perm {got['bl_kw_p_perm']:.3g}")
    check("(e) Kruskal-Wallis H and chi2 p match scipy.stats.kruskal for 3 units", ok, "; ".join(det))
    calib_kw = {}
    for sid in sids:
        u, t, al = load_session(pdir, sid)
        t = t.sort_values("trial_index").reset_index(drop=True)
        blk = t["block_index"].to_numpy(); tb = t["trial_index_in_block"].to_numpy(); sw_ = blk > 0
        Rb = per_trial_rates(al, t, u, tuple(P["baseline_window"]))
        im, em, lm, _ = masks_for(t, "both", P["instruction_n"], tuple(P["early_window"]), P["late_n"])  # local to this session
        real = kw_instruction_test(Rb, im, em, lm, blk, 500, np.random.default_rng(1))
        pi, pe, pl = sw_ & (tb >= 40) & (tb < 45), sw_ & (tb >= 45) & (tb < 55), sw_ & (tb >= 20) & (tb < 30)
        pseudo = kw_instruction_test(Rb, pi, pe, pl, blk, 500, np.random.default_rng(1))
        flag = lambda k: (k["p_perm"] < 0.05) & (k["p_early"] < 0.05) & (k["p_late"] < 0.05)  # the app's flag criterion (BH on hold)
        calib_kw[sid] = {"real_omnibus_p05": float((real["p_perm"] < 0.05).mean()), "pseudo_omnibus_p05": float((pseudo["p_perm"] < 0.05).mean()),
                         "real_flag": float(flag(real).mean()), "pseudo_flag": float(flag(pseudo).mean())}
        print(f"[info] KW {sid}: omnibus p<0.05 real {calib_kw[sid]['real_omnibus_p05']:.3f} / mid-block pseudo {calib_kw[sid]['pseudo_omnibus_p05']:.3f} | "
              f"flag (omnibus & both post-hoc p<0.05) real {calib_kw[sid]['real_flag']:.3f} / pseudo {calib_kw[sid]['pseudo_flag']:.3f}")
    ps = np.array([v["pseudo_flag"] for v in calib_kw.values()]); po = np.array([v["pseudo_omnibus_p05"] for v in calib_kw.values()])
    check("(e) KW flag null: mid-block pseudo phases flagged <= 0.05 mean, <= 0.08 max", ps.mean() <= 0.05 and ps.max() <= 0.08,
          f"pseudo flag mean {ps.mean():.3f} max {ps.max():.3f} (omnibus-only pseudo mean {po.mean():.3f}, anticonservative because trial shuffles ignore autocorrelation) | real flag mean {np.mean([v['real_flag'] for v in calib_kw.values()]):.3f}")
    summ["calibration"]["kw_pseudo_phases"] = {"pseudo_instruction": "40<=tib<45", "pseudo_early": "45<=tib<55", "pseudo_late": "20<=tib<30", "per_session": calib_kw}
    summ_path.write_text(json.dumps(summ, indent=2, default=float))
    for w in ("bl", "ev"):
        check(f"(e) {w}_kw: q_bh >= p_perm; rule_updating implies omnibus p<0.05 and both post-hoc p<0.05; candidate implies rule_updating",
              bool((table[f"{w}_kw_q_bh"] >= table[f"{w}_kw_p_perm"] - 1e-12).all()
                   and (~table[f"{w}_kw_is_rule_updating"] | ((table[f"{w}_kw_p_perm"] < 0.05) & (table[f"{w}_kw_p_early"] < 0.05) & (table[f"{w}_kw_p_late"] < 0.05))).all()
                   and (~table[f"{w}_kw_is_candidate"] | table[f"{w}_kw_is_rule_updating"]).all()))

    # (d) profiles
    counts = prof.groupby(["session_id", "unit_index", "context", "align", "window"], observed=True).size()
    exp = {"block_start": P["profile_start_n"], "block_end": P["profile_end_n"]}
    ok = all((counts.xs(al, level="align") == n).all() for al, n in exp.items())
    check("(d) every (unit, context, align, window) has the expected number of rel_trial rows", ok, f"n groups {len(counts)}")
    ui = int(picks[0])
    R_bl = per_trial_rates(aligned, trials, units, tuple(P["baseline_window"]))
    row = int(np.where(units["unit_index"].to_numpy() == ui)[0][0])
    blocks = sorted(trials.loc[sw, "block_index"].unique())
    ok = True
    for rel in (0, 5, 10):
        hand = np.mean([R_bl[row, sw & (trials["block_index"].to_numpy() == b) & (tib == rel)].mean() for b in blocks])
        got = prof[(prof["session_id"] == a.session) & (prof["unit_index"] == ui) & (prof["context"] == "both") & (prof["align"] == "block_start")
                   & (prof["window"] == "baseline") & (prof["rel_trial"] == rel)]["mean_rate"].iloc[0]
        if abs(hand - got) > 1e-4:
            ok = False
    for rel in (-1, -10):
        hand = np.mean([R_bl[row, sw & (trials["block_index"].to_numpy() == b) & (tib - nb == rel)].mean() for b in blocks])
        got = prof[(prof["session_id"] == a.session) & (prof["unit_index"] == ui) & (prof["context"] == "both") & (prof["align"] == "block_end")
                   & (prof["window"] == "baseline") & (prof["rel_trial"] == rel)]["mean_rate"].iloc[0]
        if abs(hand - got) > 1e-4:
            ok = False
    check("(d) hand-recomputed profile values match (start 0, 5, 10; end -1, -10)", ok, f"unit {ui}")

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
