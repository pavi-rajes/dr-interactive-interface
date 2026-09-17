#!/usr/bin/env python3
"""Screen units for rule-change (block-switch) responses.

Design
------
A rule-updating unit fires differently in the early trials after a rule change than in
the steady-state late part of the same block. Within every block the first INSTRUCTION_N
(5) trials are instruction trials; the early window is trials [EARLY_START, EARLY_END) =
5..14; the late window is the last LATE_N (10) trials of the block. We contrast early vs
late within switch blocks (blocks 1..5), separately per block context (the block's rewarded
modality: "aud", "vis") and pooled ("both"), in two analysis windows: baseline (-1.5 to
-0.06 s) and stimulus-evoked (0.0 to 0.5 s). A block's context equals the switch direction
that produced it (aud-context blocks follow vis_to_aud switches), so `direction` is kept as
a derived column.

Secondary screens (early vs late, instruction vs late): detrended (2-block centred moving average) within-block chunk permutation.
Trials are grouped into `chunk_size`-trial chunks within (block, label) strata so no chunk
straddles the excluded gap; early/late chunk labels are permuted within each block
(preserving counts); statistic = mean_early - mean_late. Because instruction trials are
all rewarded-modality targets, the evoked rates are first residualized on
(stim_name x rewarded_modality) per session; the un-residualized p is kept as
`ev_p_perm_raw`. Mid-block pseudo-switch calibration (see check_rule_change_units.py)
gives ~7% of units at p<0.05 (5-trial chunks) vs ~9-18% for the real early-vs-late contrast:
a modest, calibrated exploratory screen. Because BH over ~6,700 units needs p-values far below
1/2000, the default is 20,000 permutations (p floor 5e-5, ~25 s for all sessions).

Primary test (scientist's definition, 2026-09-15): a rule-updating unit fires MORE during the
instruction trials than during the rest of the block. Per unit, a Kruskal-Wallis H across the
three phases (instruction / early / late) is compared with a null distribution obtained by
shuffling the phase labels across those trials within each block (trial-number shuffle,
phase counts preserved); kw_p_perm = (1 + #(H_null >= H_obs)) / (n_perm + 1). The unit is
rule-updating if the omnibus kw_p_perm < 0.05 and both post-hoc contrasts (instruction > early,
instruction > late) have permutation p < 0.05 (`<w>_kw_is_rule_updating`). BH correction is ON
HOLD by the scientist's direction: across-units q (`<w>_kw_q_bh`) and per-unit-family q
(`<w>_kw_q_unit_*`) are computed as supplementary columns only. The chunk-permutation contrasts below (early vs late, instruction vs late) are kept
as secondary screens.

Per-unit null pass rates (`<w>_kw_null_rate_p05/p01/def`): fraction of the label shuffles under which the
unit would itself pass the flag criterion (the p01 rate runs slightly above nominal for low-rate units because
of ties in the rank statistics) (omnibus and both post-hoc at the 5% / 1% null quantiles, or the
definition proxy: instruction mean rank above early and late). Summed over a region they give the number of
units expected to be flagged under the null.

Flags
-----
<w>_kw_is_instruction_higher : DEFINITION (scientist): instruction mean > early mean AND > late mean. DEFAULT set in the viewer.
<w>_kw_is_rule_updating : optional stricter tier: omnibus kw_p_perm < 0.05 AND both one-sided post-hoc contrasts (instruction > early,
                          instruction > late; Dunn-style mean-rank differences on the same shuffles) at
                          uncorrected permutation p < 0.05; plus rate gate.        (optional tier)
<w>_kw_is_candidate     : same with 0.01 thresholds
<w>_kw_is_bh            : supplementary (BH on hold): across-units BH q of the omnibus < alpha and per-unit BH q of
                          both post-hoc contrasts < alpha (family = contexts x windows x contrasts)
<w>_is_significant : <w>_q_bh < alpha                      (BH within each direction x window)
<w>_is_robust      : significant & <w>_consistency >= min_consistency & max(mean_early, mean_late) >= min_rate
is_rule_updating   : bl_is_robust | ev_is_robust            (secondary early-vs-late screen)
rule_updating_window : baseline | evoked | both | none
With 2-3 switch blocks per context, min_consistency 0.8 requires every block of that
context to agree in sign.

Outputs (in --out-dir): rule_change_units.parquet (3 rows per unit: context aud/vis/both), rule_change_profiles.parquet (block_start / block_end aligned),
rule_change_profiles_by_region.parquet, rule_change_by_region.parquet,
rule_change_by_region_session.parquet, rule_change_summary.json
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
from analyze_context_baseline_shift import contrast_weights, detrend_rates, load_session, make_chunks  # noqa: E402

SCRIPT_VERSION = "2.0"
UNIT_META = ["session_id", "unit_index", "unit_id", "structure", "location", "ccf_ap", "ccf_dv", "ccf_ml", "task_firing_rate"]
CONTEXTS = ["aud", "vis", "both"]
DIRECTION_OF = {"aud": "vis_to_aud", "vis": "aud_to_vis", "both": "both"}
WINDOWS = {"bl": "baseline", "ev": "evoked"}


# ----------------------------------------------------------------------------- rates
def per_trial_rates(aligned: pd.DataFrame, trials: pd.DataFrame, units: pd.DataFrame, window) -> np.ndarray:
    """Dense (units x all trials) matrix of firing rate (Hz) in the window, trials in trial_index order."""
    start, end = window
    a = aligned[(aligned["t_rel"] >= start) & (aligned["t_rel"] < end)]
    counts = a.groupby(["unit_index", "trial_index"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=units["unit_index"].to_numpy(), columns=trials["trial_index"].to_numpy(), fill_value=0)
    return counts.to_numpy(dtype=float) / (end - start)


def residualize_on_stimulus(R: np.ndarray, trials: pd.DataFrame) -> np.ndarray:
    """Subtract per-unit session mean for each (stim_name, rewarded_modality) cell."""
    cell = (trials["stim_name"].astype(str) + "|" + trials["rewarded_modality"].astype(str)).to_numpy()
    out = R.copy()
    for c in np.unique(cell):
        m = cell == c
        out[:, m] = R[:, m] - R[:, m].mean(axis=1, keepdims=True)
    return out


# ----------------------------------------------------------------------------- test
def early_late_test(R_d: np.ndarray, early_mask: np.ndarray, late_mask: np.ndarray, block: np.ndarray,
                    chunk_size: int, n_perm: int, rng: np.random.Generator):
    """Detrended within-block chunk permutation of early(1)/late(0) labels.

    Returns (obs_diff, p_perm, n_chunks_early, n_chunks_late). Chunks are formed within
    (block, label) strata; labels are permuted within each block preserving counts."""
    sel = early_mask | late_mask
    Rs = R_d[:, sel]
    blk = block[sel]
    label = early_mask[sel].astype(int)
    cid = make_chunks(blk * 2 + label, chunk_size)
    n_chunks = cid.max() + 1
    chunk_label = np.empty(n_chunks, dtype=int)
    chunk_block = np.empty(n_chunks, dtype=int)
    for c in range(n_chunks):
        m = cid == c
        labs = np.unique(label[m])
        assert len(labs) == 1, "chunk straddles early/late labels"
        chunk_label[c] = labs[0]
        chunk_block[c] = blk[m][0]
    obs = Rs @ contrast_weights(label)
    W = np.empty((Rs.shape[1], n_perm))
    for k in range(n_perm):
        perm = chunk_label.copy()
        for b in np.unique(chunk_block):
            m = chunk_block == b
            perm[m] = rng.permutation(chunk_label[m])
        W[:, k] = contrast_weights(perm[cid])
    P = Rs @ W
    p = (1 + (np.abs(P) >= np.abs(obs)[:, None]).sum(axis=1)) / (n_perm + 1)
    return obs, p, int(chunk_label.sum()), int(n_chunks - chunk_label.sum())


def kw_instruction_test(R: np.ndarray, instruction: np.ndarray, early: np.ndarray, late: np.ndarray, block: np.ndarray,
                        n_perm: int, rng: np.random.Generator, batch: int = 1000):
    """Kruskal-Wallis H (tie-corrected) across instruction / early / late trials for every unit, with a
    trial-label-shuffle null within each block, plus Dunn-style one-sided post-hoc contrasts on the same
    shuffles (mean rank of instruction trials minus mean rank of early / late trials).

    Returns dict with H_obs, p_perm, p_chi2, d_early, p_early, d_late, p_late."""
    from scipy.stats import chi2, rankdata
    sel = instruction | early | late
    Rs = R[:, sel]
    labels = np.where(instruction[sel], 0, np.where(early[sel], 1, 2))
    blk = block[sel]
    N = Rs.shape[1]
    ranks = rankdata(Rs, axis=1)
    C = np.ones(len(Rs))
    for i in range(len(Rs)):
        _, t = np.unique(Rs[i], return_counts=True)
        C[i] = 1 - (t.astype(float) ** 3 - t).sum() / (N ** 3 - N)
    n_g = np.bincount(labels, minlength=3).astype(float)

    def stats_from(W: np.ndarray):
        """W: (N, 3k) one-hot labels -> H (units, k), d_early (units, k), d_late (units, k)."""
        k = W.shape[1] // 3
        S = (ranks @ W).reshape(len(Rs), k, 3)
        H = 12.0 / (N * (N + 1)) * (S ** 2 / n_g).sum(axis=2) - 3 * (N + 1)
        mean_rank = S / n_g
        return H, mean_rank[:, :, 0] - mean_rank[:, :, 1], mean_rank[:, :, 0] - mean_rank[:, :, 2]

    def onehot(lab: np.ndarray, k_off: int, W: np.ndarray):
        W[np.arange(N), 3 * k_off + lab] = 1.0

    W0 = np.zeros((N, 3)); onehot(labels, 0, W0)
    H_obs, de_obs, dl_obs = (x[:, 0] for x in stats_from(W0))
    with np.errstate(invalid="ignore", divide="ignore"):
        H_obs = np.where(C > 0, H_obs / C, np.nan)
    cnt_h, cnt_e, cnt_l = np.zeros(len(Rs)), np.zeros(len(Rs)), np.zeros(len(Rs))
    Hn_all = np.empty((len(Rs), n_perm), dtype=np.float32)
    den_all = np.empty((len(Rs), n_perm), dtype=np.float32)
    dln_all = np.empty((len(Rs), n_perm), dtype=np.float32)
    blocks = np.unique(blk)
    done = 0
    while done < n_perm:
        k = min(batch, n_perm - done)
        W = np.zeros((N, 3 * k))
        for j in range(k):
            perm = labels.copy()
            for b in blocks:
                m = blk == b
                perm[m] = rng.permutation(labels[m])
            onehot(perm, j, W)
        Hn, den, dln = stats_from(W)
        with np.errstate(invalid="ignore", divide="ignore"):
            Hn = Hn / C[:, None]
        cnt_h += (Hn >= H_obs[:, None]).sum(axis=1)
        cnt_e += (den >= de_obs[:, None]).sum(axis=1)
        cnt_l += (dln >= dl_obs[:, None]).sum(axis=1)
        Hn_all[:, done:done + k], den_all[:, done:done + k], dln_all[:, done:done + k] = Hn, den, dln
        done += k
    # per-unit null pass rates of the flag criteria (fraction of shuffles that would be flagged):
    # a shuffle passes the omnibus at level a if its H exceeds the (1-a) quantile of the null, likewise post-hoc.
    null_rate = {}
    for a_, name in ((0.05, "p05"), (0.01, "p01")):
        qh = np.nanquantile(Hn_all, 1 - a_, axis=1)[:, None]
        qe = np.nanquantile(den_all, 1 - a_, axis=1)[:, None]
        ql = np.nanquantile(dln_all, 1 - a_, axis=1)[:, None]
        null_rate[name] = ((Hn_all >= qh) & (den_all >= qe) & (dln_all >= ql)).mean(axis=1)
    null_rate["def"] = ((den_all > 0) & (dln_all > 0)).mean(axis=1)  # definition proxy: instruction mean rank above both
    nan = np.isnan(H_obs)
    out = {
        "H_obs": H_obs, "p_perm": np.where(nan, 1.0, (1 + cnt_h) / (n_perm + 1)),
        "p_chi2": np.where(nan, 1.0, chi2.sf(np.nan_to_num(H_obs), df=2)),
        "d_early": de_obs, "p_early": np.where(nan, 1.0, (1 + cnt_e) / (n_perm + 1)),
        "d_late": dl_obs, "p_late": np.where(nan, 1.0, (1 + cnt_l) / (n_perm + 1)),
        "null_rate_p05": null_rate["p05"], "null_rate_p01": null_rate["p01"], "null_rate_def": null_rate["def"],
    }
    return out


def block_lengths(trials: pd.DataFrame) -> np.ndarray:
    """Number of trials in each trial's block (array aligned with trials)."""
    n = trials.groupby("block_index")["trial_index"].transform("size")
    return n.to_numpy()


def masks_for(trials: pd.DataFrame, context: str, instruction_n: int, early_window: tuple[int, int], late_n: int,
              include_first_block: bool = False):
    """Return (instruction, early, late masks, blocks) for blocks of the given context ('aud', 'vis', 'both')."""
    in_ctx = (trials["block_index"] > 0) | include_first_block
    if context != "both":
        in_ctx &= trials["rewarded_modality"].astype(str) == context
    in_ctx = in_ctx.to_numpy()
    tib = trials["trial_index_in_block"].to_numpy()
    nb = block_lengths(trials)
    instruction = in_ctx & (tib < instruction_n)
    early = in_ctx & (tib >= early_window[0]) & (tib < early_window[1])
    late = in_ctx & (tib >= nb - late_n)
    blocks = sorted(trials.loc[in_ctx, "block_index"].unique().tolist())
    return instruction, early, late, blocks


def unit_context_stats(mats: dict, trials: pd.DataFrame, units: pd.DataFrame, context: str, a, rng) -> pd.DataFrame:
    """One row per unit for a context. mats: {'bl', 'ev', 'bl_d', 'ev_d', 'ev_res_d'}"""
    block = trials["block_index"].to_numpy()
    instruction, early, late, blocks = masks_for(trials, context, a.instruction_n, tuple(a.early_window), a.late_n)
    out = units[UNIT_META].copy().reset_index(drop=True)
    out["context"] = context
    out["direction"] = DIRECTION_OF[context]
    out["n_blocks"] = len(blocks)
    out["n_trials_instruction"], out["n_trials_early"], out["n_trials_late"] = int(instruction.sum()), int(early.sum()), int(late.sum())
    for w in ("bl", "ev"):
        R = mats[w]
        mi = R[:, instruction].mean(axis=1) if instruction.any() else np.full(len(R), np.nan)
        ml = R[:, late].mean(axis=1)
        out[f"{w}_mean_instruction"], out[f"{w}_mean_late"] = mi, ml
        out[f"{w}_mean_early"] = R[:, early].mean(axis=1)
        for cpre, first_mask, first_name in (("", early, "early"), ("inst_", instruction, "instruction")):
            mf = R[:, first_mask].mean(axis=1)
            diff = mf - ml
            out[f"{w}_{cpre}diff"] = diff
            out[f"{w}_{cpre}log2_ratio"] = np.log2((mf + 0.1) / (ml + 0.1))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{w}_{cpre}modulation_index"] = np.where(mf + ml > 0, (mf - ml) / (mf + ml), np.nan)
            out[f"{w}_{cpre}direction_sign"] = np.where(diff > 0, f"{first_name}_higher", np.where(diff < 0, "late_higher", "none"))
            sign = np.sign(diff)
            matches = []
            for b in blocks:
                fb, lb = first_mask & (block == b), late & (block == b)
                bd = R[:, fb].mean(axis=1) - R[:, lb].mean(axis=1)
                out[f"{w}_{cpre}block_diff_{b}"] = bd
                matches.append(np.sign(bd) == sign)
            cons = np.mean(np.stack(matches, axis=1), axis=1) if matches else np.full(len(R), np.nan)
            out[f"{w}_{cpre}consistency"] = np.where(sign == 0, np.nan, cons)
            R_test = mats["ev_res_d"] if w == "ev" else mats["bl_d"]
            obs, pp, ncf, ncl = early_late_test(R_test, first_mask, late, block, a.chunk_size, a.n_perm, rng)
            out[f"{w}_{cpre}p_perm"] = pp
            out[f"{w}_{cpre}n_chunks_first"], out[f"{w}_{cpre}n_chunks_late"] = ncf, ncl
            if w == "ev":
                out[f"ev_{cpre}obs_diff_residual"] = obs
                _, p_raw, _, _ = early_late_test(mats["ev_d"], first_mask, late, block, a.chunk_size, a.n_perm, rng)
                out[f"ev_{cpre}p_perm_raw"] = p_raw
            else:
                out[f"bl_{cpre}obs_diff_detrended"] = obs
        # primary: Kruskal-Wallis instruction / early / late with trial-label shuffle null (raw rates) + Dunn post-hoc
        kw = kw_instruction_test(R, instruction, early, late, block, a.n_perm_kw, rng)
        rest = R[:, early | late].mean(axis=1)
        out[f"{w}_kw_h"], out[f"{w}_kw_p_perm"], out[f"{w}_kw_p_chi2"] = kw["H_obs"], kw["p_perm"], kw["p_chi2"]
        out[f"{w}_kw_d_early"], out[f"{w}_kw_p_early"] = kw["d_early"], kw["p_early"]   # instruction > early (one-sided)
        out[f"{w}_kw_d_late"], out[f"{w}_kw_p_late"] = kw["d_late"], kw["p_late"]       # instruction > late  (one-sided)
        out[f"{w}_mean_rest"] = rest
        out[f"{w}_kw_null_rate_p05"], out[f"{w}_kw_null_rate_p01"], out[f"{w}_kw_null_rate_def"] = kw["null_rate_p05"], kw["null_rate_p01"], kw["null_rate_def"]
        out[f"{w}_kw_log2_ratio"] = np.log2((mi + 0.1) / (rest + 0.1))
        out[f"{w}_kw_instruction_higher"] = (mi > out[f"{w}_mean_early"].to_numpy()) & (mi > ml)
        out[f"{w}_kw_direction_sign"] = np.where(mi > rest, "instruction_higher", np.where(mi < rest, "rest_higher", "none"))
        if w == "ev":  # reference: same test on stimulus-residualized evoked rates
            kwr = kw_instruction_test(mats["ev_res"], instruction, early, late, block, a.n_perm_kw, rng)
            out["ev_kw_p_perm_residualized"] = kwr["p_perm"]
    return out


def add_flags(table: pd.DataFrame, alpha: float, min_consistency: float, min_rate: float, p_candidate: float = 0.01,
              p_liberal: float = 0.05) -> pd.DataFrame:
    """Tiered flags per window x contrast: is_robust (BH q<alpha), is_candidate (p<p_candidate), is_liberal (p<p_liberal),
    each also requiring block consistency and a minimum rate."""
    table = table.copy()
    for cpre in ("", "inst_"):
        first = "early" if cpre == "" else "instruction"
        for w in ("bl", "ev"):
            q = np.full(len(table), np.nan)
            for c in CONTEXTS:
                m = (table["context"] == c).to_numpy()
                q[m] = stats.false_discovery_control(table.loc[m, f"{w}_{cpre}p_perm"].to_numpy(), method="bh")
            table[f"{w}_{cpre}q_bh"] = q
            gate = ((table[f"{w}_{cpre}consistency"] >= min_consistency)
                    & (np.maximum(table[f"{w}_mean_{first}"], table[f"{w}_mean_late"]) >= min_rate)).fillna(False)
            table[f"{w}_{cpre}is_significant"] = table[f"{w}_{cpre}q_bh"] < alpha
            table[f"{w}_{cpre}is_robust"] = (table[f"{w}_{cpre}is_significant"] & gate).astype(bool)
            table[f"{w}_{cpre}is_candidate"] = ((table[f"{w}_{cpre}p_perm"] < p_candidate) & gate).astype(bool)
            table[f"{w}_{cpre}is_liberal"] = ((table[f"{w}_{cpre}p_perm"] < p_liberal) & gate).astype(bool)
        for tier in ("is_robust", "is_candidate", "is_liberal"):
            table[f"{cpre}{tier}".replace("is_robust", "is_rule_updating") if tier == "is_robust" else f"{cpre}{tier}"] = \
                table[f"bl_{cpre}{tier}"] | table[f"ev_{cpre}{tier}"]
        table[f"{cpre}rule_updating_window"] = np.select(
            [table[f"bl_{cpre}is_robust"] & table[f"ev_{cpre}is_robust"], table[f"bl_{cpre}is_robust"], table[f"ev_{cpre}is_robust"]],
            ["both", "baseline", "evoked"], default="none")
        table[f"{cpre}candidate_window"] = np.select(
            [table[f"bl_{cpre}is_candidate"] & table[f"ev_{cpre}is_candidate"], table[f"bl_{cpre}is_candidate"], table[f"ev_{cpre}is_candidate"]],
            ["both", "baseline", "evoked"], default="none")
    # primary KW flags. Per-unit BH over the family of post-hoc permutation p-values:
    # contexts (aud, vis, both) x windows (bl, ev) x contrasts (instruction>early, instruction>late) = 12 tests per unit.
    fam_cols = [f"{w}_kw_p_{c}" for w in ("bl", "ev") for c in ("early", "late")]
    wide = table.pivot_table(index=["session_id", "unit_index"], columns="context", values=fam_cols)
    qwide = wide.copy()
    P_ = wide.to_numpy()
    Q_ = np.empty_like(P_)
    for i in range(P_.shape[0]):
        Q_[i] = stats.false_discovery_control(P_[i], method="bh")
    qwide[:] = Q_
    qlong = qwide.stack(level="context", future_stack=True).reset_index()
    qlong = qlong.rename(columns={c: c.replace("_kw_p_", "_kw_q_unit_") for c in fam_cols})
    table = table.merge(qlong, on=["session_id", "unit_index", "context"], how="left")
    for w in ("bl", "ev"):
        q = np.full(len(table), np.nan)  # across-units BH on the omnibus (supplementary)
        for c in CONTEXTS:
            m = (table["context"] == c).to_numpy()
            q[m] = stats.false_discovery_control(table.loc[m, f"{w}_kw_p_perm"].to_numpy(), method="bh")
        table[f"{w}_kw_q_bh"] = q
        gate = np.maximum(table[f"{w}_mean_instruction"], table[f"{w}_mean_rest"]) >= min_rate
        # BH ON HOLD (scientist, 2026-09-15): flags use uncorrected permutation p-values; q columns are supplementary.
        posthoc = (table[f"{w}_kw_p_early"] < p_liberal) & (table[f"{w}_kw_p_late"] < p_liberal)
        table[f"{w}_kw_is_rule_updating"] = ((table[f"{w}_kw_p_perm"] < p_liberal) & posthoc & gate).astype(bool)
        table[f"{w}_kw_is_candidate"] = ((table[f"{w}_kw_p_perm"] < p_candidate) & (table[f"{w}_kw_p_early"] < p_candidate)
                                         & (table[f"{w}_kw_p_late"] < p_candidate) & gate).astype(bool)
        table[f"{w}_kw_is_bh"] = ((table[f"{w}_kw_q_bh"] < alpha) & (table[f"{w}_kw_q_unit_early"] < alpha)
                                  & (table[f"{w}_kw_q_unit_late"] < alpha) & gate).astype(bool)
    # DEFINITION (scientist, 2026-09-15): a rule-updating unit has mean rate in instruction trials > early AND > late.
    # No statistical filter; the tests below are optional stricter tiers.
    for w in ("bl", "ev"):
        table[f"{w}_kw_is_instruction_higher"] = table[f"{w}_kw_instruction_higher"].astype(bool)
    for tier in ("is_instruction_higher", "is_rule_updating", "is_candidate", "is_bh"):
        table[f"kw_{tier}"] = table[f"bl_kw_{tier}"] | table[f"ev_kw_{tier}"]
    table["instruction_higher_window"] = np.select([table["bl_kw_is_instruction_higher"] & table["ev_kw_is_instruction_higher"],
                                                    table["bl_kw_is_instruction_higher"], table["ev_kw_is_instruction_higher"]],
                                                   ["both", "baseline", "evoked"], default="none")
    table["kw_window"] = np.select([table["bl_kw_is_rule_updating"] & table["ev_kw_is_rule_updating"], table["bl_kw_is_rule_updating"],
                                    table["ev_kw_is_rule_updating"]], ["both", "baseline", "evoked"], default="none")
    return table


# ----------------------------------------------------------------------------- profiles
def block_profiles(mats: dict, trials: pd.DataFrame, units: pd.DataFrame, start_n: int, end_n: int) -> pd.DataFrame:
    """Mean raw rate per within-block position, averaged over switch blocks of a context.

    align == 'block_start': rel_trial = trial_index_in_block (0 .. start_n-1)
    align == 'block_end'  : rel_trial = trial_index_in_block - block_length (-end_n .. -1)"""
    rows = []
    tib = trials["trial_index_in_block"].to_numpy()
    rel_end = tib - block_lengths(trials)
    block = trials["block_index"].to_numpy()
    ctx_of_trial = trials["rewarded_modality"].astype(str).to_numpy()
    for context in CONTEXTS:
        in_ctx = (block > 0) & ((ctx_of_trial == context) if context != "both" else True)
        blocks = np.unique(block[in_ctx])
        for align, positions, rel in (("block_start", np.arange(start_n), tib), ("block_end", np.arange(-end_n, 0), rel_end)):
            for w, wname in WINDOWS.items():
                R = mats[w]
                acc = np.zeros((len(units), len(positions)))
                cnt = np.zeros(len(positions))
                for b in blocks:
                    for j, pos in enumerate(positions):
                        m = in_ctx & (block == b) & (rel == pos)
                        if m.any():
                            acc[:, j] += R[:, m].mean(axis=1)
                            cnt[j] += 1
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = acc / cnt
                rows.append(pd.DataFrame({
                    "session_id": np.repeat(units["session_id"].to_numpy(), len(positions)),
                    "unit_index": np.repeat(units["unit_index"].to_numpy(), len(positions)).astype(np.int32),
                    "context": context, "align": align, "window": wname,
                    "rel_trial": np.tile(positions, len(units)).astype(np.int16),
                    "mean_rate": mean.ravel().astype(np.float32),
                    "n_blocks": np.tile(cnt, len(units)).astype(np.int8),
                }))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- region summary
def region_summary(both: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        ru = g[g["is_rule_updating"]]
        cand = g[g["is_rule_updating"] | g["is_candidate"]]
        use_ev = cand["ev_is_robust"] | (cand["ev_is_candidate"] & ~cand["bl_is_candidate"])
        sign = np.where(use_ev, cand["ev_direction_sign"], cand["bl_direction_sign"])
        lr = np.where(use_ev, cand["ev_log2_ratio"], cand["bl_log2_ratio"])
        return pd.Series({
            "n_units": len(g), "n_instruction_higher": int(g["kw_is_instruction_higher"].sum()), "frac_instruction_higher": g["kw_is_instruction_higher"].mean(),
            "n_kw_rule_updating": int(g["kw_is_rule_updating"].sum()), "frac_kw_rule_updating": g["kw_is_rule_updating"].mean(),
            "n_kw_candidate": int(g["kw_is_candidate"].sum()), "n_kw_bh": int(g["kw_is_bh"].sum()),
            "n_rule_updating": int(len(ru)), "frac_rule_updating": len(ru) / len(g),
            "n_candidate": int(g["is_candidate"].sum()), "frac_candidate": g["is_candidate"].mean(),
            "n_inst_candidate": int(g["inst_is_candidate"].sum()), "frac_inst_candidate": g["inst_is_candidate"].mean(),
            "n_baseline_robust": int(g["bl_is_robust"].sum()), "n_evoked_robust": int(g["ev_is_robust"].sum()),
            "n_early_higher": int((sign == "early_higher").sum()), "n_late_higher": int((sign == "late_higher").sum()),
            "median_abs_log2_ratio": float(np.median(np.abs(lr))) if len(cand) else np.nan,
            "n_sessions": g["session_id"].nunique() if "session_id" in g.columns else 1,
        })
    out = both.groupby(by, sort=True).apply(agg, include_groups=False).reset_index()
    if by == ["structure"]:
        out = out.sort_values(["frac_instruction_higher", "n_units"], ascending=[False, False])
    else:
        out = out.drop(columns="n_sessions")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precomputed-dir", type=Path, default=config.PRECOMPUTED_DIR)
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--baseline-window", nargs=2, type=float, default=(-1.5, -0.06))
    ap.add_argument("--evoked-window", nargs=2, type=float, default=(0.0, 0.5))
    ap.add_argument("--instruction-n", type=int, default=5, help="first N trials of a block are instruction trials")
    ap.add_argument("--early-window", nargs=2, type=int, default=(5, 15), metavar=("START", "END"), help="early = START <= trial_index_in_block < END")
    ap.add_argument("--late-n", type=int, default=10, help="late = last N trials of the block")
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--detrend-window-blocks", type=float, default=2.0)
    ap.add_argument("--n-perm", type=int, default=20000, help="permutations; 20000 gives a p floor of 5e-5 so BH over ~6700 units can resolve")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-consistency", type=float, default=0.8)
    ap.add_argument("--min-rate", type=float, default=0.1)
    ap.add_argument("--n-perm-kw", type=int, default=5000, help="trial-label shuffles for the Kruskal-Wallis null")
    ap.add_argument("--p-candidate", type=float, default=0.01, help="uncorrected p threshold for the candidate tier")
    ap.add_argument("--p-liberal", type=float, default=0.05, help="uncorrected p threshold for the liberal tier")
    ap.add_argument("--profile-start-n", type=int, default=50, help="block_start-aligned profile length")
    ap.add_argument("--profile-end-n", type=int, default=20, help="block_end-aligned profile length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=config.PRECOMPUTED_DIR)
    a = ap.parse_args()

    sids = a.sessions or sorted(p.parent.name for p in a.precomputed_dir.glob("*/aligned_spikes.parquet"))
    t0 = time.time()
    tables, profiles = [], []
    for sid in sids:
        ts = time.time()
        rng = np.random.default_rng(a.seed)
        units, trials, aligned = load_session(a.precomputed_dir, sid)
        trials = trials.sort_values("trial_index").reset_index(drop=True)
        block = trials["block_index"].to_numpy()
        R_bl = per_trial_rates(aligned, trials, units, tuple(a.baseline_window))
        R_ev = per_trial_rates(aligned, trials, units, tuple(a.evoked_window))
        R_ev_res = residualize_on_stimulus(R_ev, trials)
        mats = {"bl": R_bl, "ev": R_ev, "ev_res": R_ev_res,
                "bl_d": detrend_rates(R_bl, block, a.detrend_window_blocks),
                "ev_d": detrend_rates(R_ev, block, a.detrend_window_blocks),
                "ev_res_d": detrend_rates(R_ev_res, block, a.detrend_window_blocks)}
        for c in CONTEXTS:
            tables.append(unit_context_stats(mats, trials, units, c, a, rng))
        profiles.append(block_profiles(mats, trials, units, a.profile_start_n, a.profile_end_n))
        print(f"[ok] {sid}: {len(units)} units, {len(trials)} trials, {time.time() - ts:.1f}s")

    table = add_flags(pd.concat(tables, ignore_index=True), a.alpha, a.min_consistency, a.min_rate, a.p_candidate, a.p_liberal)
    table = table.sort_values(["session_id", "unit_index", "context"]).reset_index(drop=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(a.out_dir / "rule_change_units.parquet", index=False)
    prof = pd.concat(profiles, ignore_index=True)
    prof.to_parquet(a.out_dir / "rule_change_profiles.parquet", index=False, compression="zstd")

    both = table[table["context"] == "both"]
    by_region = region_summary(both, ["structure"])
    by_region.to_parquet(a.out_dir / "rule_change_by_region.parquet", index=False)
    region_summary(both, ["session_id", "structure"]).to_parquet(a.out_dir / "rule_change_by_region_session.parquet", index=False)
    # population profiles: rule-updating units (definition: instruction > early and > late) per structure with >= 20 units total
    big = by_region.loc[by_region["n_units"] >= 20, "structure"]
    ru_keys = both.loc[both["kw_is_instruction_higher"] & both["structure"].isin(big), ["session_id", "unit_index", "structure"]]
    pop = prof.merge(ru_keys, on=["session_id", "unit_index"]).groupby(["structure", "context", "align", "window", "rel_trial"], observed=True)
    pop = pop.agg(mean_rate=("mean_rate", "mean"), sem_rate=("mean_rate", "sem"), n_units=("unit_index", "size")).reset_index()
    pop.to_parquet(a.out_dir / "rule_change_profiles_by_region.parquet", index=False)

    n_units = len(both)
    tiers = ["kw_is_instruction_higher", "kw_is_rule_updating", "kw_is_candidate", "kw_is_bh", "is_rule_updating", "is_candidate", "is_liberal", "inst_is_rule_updating", "inst_is_candidate", "inst_is_liberal"]
    per_ctx = table.groupby("context")[tiers].sum().astype(int)
    per_session = both.groupby("session_id")[tiers].sum().astype(int)
    top = by_region[by_region["n_units"] >= 20].head(15)
    counts = {t: int(both[t].sum()) for t in tiers}
    print(f"\nUnits {n_units} (context=both). DEFINITION instruction > early and > late: {counts['kw_is_instruction_higher']} ({counts['kw_is_instruction_higher'] / n_units:.1%}); "
          f"with KW omnibus + post-hoc p<{a.p_liberal}: {counts['kw_is_rule_updating']}, p<{a.p_candidate}: {counts['kw_is_candidate']}, BH q<{a.alpha}: {counts['kw_is_bh']} "
          f"| by window {both['kw_window'].value_counts().to_dict()}")
    print(f"Secondary chunk-permutation contrasts. early-vs-late: robust(BH) {counts['is_rule_updating']}, candidate(p<{a.p_candidate}) "
          f"{counts['is_candidate']}, liberal(p<{a.p_liberal}) {counts['is_liberal']} | instruction-vs-late: robust {counts['inst_is_rule_updating']}, "
          f"candidate {counts['inst_is_candidate']}, liberal {counts['inst_is_liberal']}")
    print("\nPer context:\n" + per_ctx.to_string())
    print("\nPer session (context=both):\n" + per_session.to_string())
    print("\nTop structures (n>=20):\n" + top[["structure", "n_units", "n_instruction_higher", "frac_instruction_higher", "n_kw_rule_updating", "frac_kw_rule_updating", "n_kw_candidate",
          "n_candidate", "n_inst_candidate", "n_sessions"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    n_ru = counts["is_rule_updating"]
    win_counts = both["rule_updating_window"].value_counts().to_dict()
    per_dir = per_ctx
    runtime = time.time() - t0
    print(f"\nruntime {runtime:.1f}s")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "script_version": SCRIPT_VERSION,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items() if k not in ("precomputed_dir", "out_dir", "sessions")},
        "flag_semantics": {
            "kw_is_instruction_higher": "DEFINITION and DEFAULT set for tab-2 (context=='both' rows): mean rate in instruction trials > early AND > late, in the baseline or evoked window; no statistical filter",
            "kw_is_rule_updating": "optional stricter tier: KW omnibus (trial-shuffle null) p<p_liberal AND one-sided post-hoc instruction>early and instruction>late at permutation p<p_liberal, rate gate (bl | ev). BH on hold.",
            "kw_is_candidate": "same with 0.01 thresholds", "kw_is_bh": "supplementary: across-units BH q (omnibus) and per-unit BH q (post-hoc, 12-test family) < alpha",
            "is_rule_updating": "secondary: early-vs-late chunk permutation, BH-robust (bl_is_robust | ev_is_robust)",
            "<w>_is_robust": "<w>_q_bh<alpha (BH within context x window) & <w>_consistency>=min_consistency & max(mean_early,mean_late)>=min_rate",
            "windows": "instruction = trial_index_in_block < instruction_n; early = early_window[0] <= tib < early_window[1]; late = last late_n trials of the block",
            "tiers": "is_rule_updating = BH-robust (q<alpha); is_candidate = p_perm<p_candidate; is_liberal = p_perm<p_liberal; all require consistency>=min_consistency and rate>=min_rate",
            "contrasts": "'kw_' = instruction vs early and late (PRIMARY, Kruskal-Wallis + post-hoc); '' prefix = early vs late (secondary chunk permutation); 'inst_' = instruction vs late (secondary)",
            "ev_p_perm": "evoked test on rates residualized on stim_name x rewarded_modality (ev_p_perm_raw = un-residualized)",
            "rule_updating_window": "baseline | evoked | both | none",
        },
        "n_units": n_units, "n_instruction_higher": counts["kw_is_instruction_higher"], "n_kw_rule_updating": counts["kw_is_rule_updating"], "n_rule_updating": n_ru, "tier_counts_both": counts, "rule_updating_window_counts": {str(k): int(v) for k, v in win_counts.items()},
        "per_context": per_dir.reset_index().to_dict(orient="records"),
        "per_session": per_session.reset_index().to_dict(orient="records"),
        "top_structures": top.to_dict(orient="records"), "runtime_s": round(runtime, 1),
    }
    (a.out_dir / "rule_change_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
