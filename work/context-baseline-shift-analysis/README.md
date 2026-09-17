---
slug: context-baseline-shift-analysis
status: done
plan_review_attempts: 3
---

# Goal
Identify units whose baseline (pre-stimulus) firing rate depends on task context. Using the precomputed aligned spikes and trials tables, compute per-trial baseline firing rate in the quiescent/pre-stimulus window (e.g. the last 0.5 to 1.0 s before stim onset, excluding the first few trials after a block switch so the measure reflects steady-state context). Compare aud-rewarded vs vis-rewarded blocks per unit with a nonparametric test (Mann-Whitney U or a permutation test that shuffles block labels at the block level to respect block structure) and compute an effect size (e.g. log2 ratio of mean rates and a modulation index). Apply multiple-comparison correction (Benjamini-Hochberg) across units.

Output `data/precomputed/context_baseline_shifts.parquet` with one row per unit: session, unit_id, structure, CCF coords, mean baseline rate aud, mean baseline rate vis, effect size, raw and corrected p-values, a boolean significance flag, and shift direction. Also output a per-structure summary (fraction of significant units per region, n units) as `data/precomputed/context_baseline_shifts_by_region.parquet`. Script lives at `code/analyze_context_baseline_shift.py`, parameters (window, alpha, exclusion of post-switch trials) exposed as CLI arguments with recorded defaults.

# Instructions

Paths relative to `/Users/pavir/Code/dr-interactive-interface`; interpreter `/opt/miniconda3/bin/python3` (pandas, pyarrow, scipy 1.17, numpy, matplotlib 3.10 installed). Inputs come from [precompute-aligned-spikes](../precompute-aligned-spikes/README.md): for each session `data/precomputed/<session_id>/{units,trials,aligned_spikes}.parquet` (schema in `code/README.md`), and `data/precomputed/units_all.parquet`. Use `code/config.py` for `PRECOMPUTED_DIR`. Relevant facts: the quiescent (no-lick) period spans about -1.53 to -0.06 s before stim onset; each session has 6 blocks of 86 to 93 trials alternating `rewarded_modality` aud/vis; `is_instruction` marks the first 5 trials of every block; `is_opto` is all False; `aligned_spikes.parquet` has `unit_index`, `trial_index`, `t_rel` in [-2, 3] s. One unit has NaN CCF coordinates (keep it; the viewer drops it when plotting).

Statistical design (why): trial-level baseline rates are strongly autocorrelated within a block and drift slowly across the session, so both a trial-level Mann-Whitney test and a naive chunk-permutation test are anticonservative. Prototyping on session 759434 showed: trial-level MWU flags about 57% of units with real labels and about 44% under block-shuffled labels; a 10-trial chunk permutation on raw rates still flags about 38% of units on a pure null (two same-context blocks labelled against each other). The primary test is therefore a **stratified chunk permutation test on detrended rates**: each unit's per-trial rate vector is detrended with a centred moving average whose window is two blocks of selected trials (about 160 trials, truncated at the session edges), then trials are grouped into contiguous 20-trial chunks within blocks and chunk labels are permuted. With this the pure null gives about 4% (max 6%) of units at p < 0.05 while the real labelling keeps about 45%. Mann-Whitney p is kept only as a descriptive column. `is_robust` (permutation-significant after BH, block-consistent, above a minimum rate) is the default flag for the tab 1 dots; `is_significant` is the liberal flag. Reviewer prototypes for the detrend and pure-null code are in the session scratchpad (`proto.py`, `alts.py`, `alts2.py`) and may be reused.

1. **Write `code/analyze_context_baseline_shift.py`** with argparse: `--precomputed-dir` (default `config.PRECOMPUTED_DIR`), `--sessions SID ...` (default all sessions with the three parquet files), `--baseline-window START END` (default `-1.5 -0.06`), `--exclude-first-n N` (default `10`; drops the first N trials of every block, which subsumes the 5 instruction trials), `--chunk-size` (default `20` trials), `--detrend-window-blocks` (default `2`; `0` disables detrending), `--n-perm` (default `2000`), `--min-rate HZ` (default `0.1`), `--alpha` (default `0.05`), `--min-consistency` (default `0.8`), `--seed 0`, `--out-dir` (default `config.PRECOMPUTED_DIR`). Functions: `per_trial_baseline_rates(aligned, trials, units, window) -> DataFrame[unit_index, trial_index, rate]`, `detrend_rates(R, block_of_trial, window_blocks) -> R_detrended` (subtract, per unit, a centred moving average of width `round(window_blocks * mean selected trials per block)` computed over the selected trials in temporal order, truncated at the edges; identity when `window_blocks == 0`), `chunk_permutation_test(R_detrended, trials_sel, chunk_size, n_perm, rng) -> (obs_diff_detrended, p_perm)`, `unit_stats(rates, trials, ...) -> DataFrame`, `region_summary(unit_table) -> DataFrame`, `main()`. Module docstring states the design and flag semantics above.

2. **Per-trial baseline rates.** For each session load the three tables. Select trials with `~is_instruction & ~is_opto & trial_index_in_block >= exclude_first_n`. Filter `aligned_spikes` to `t_rel >= START & t_rel < END`, count spikes per (`unit_index`, `trial_index`) via `groupby(...).size()`, reindex onto the full unit x selected-trial grid (fill 0), divide by `END - START` to get Hz. Keep this as a dense matrix `R` of shape (n_units, n_selected_trials) (about 2000 x 480 max) for the permutation step.

3. **Per-unit statistics** (`unit_stats`). With `ctx` = 1 for aud, 0 for vis over the selected trials:
   - Descriptives: `n_trials_aud`, `n_trials_vis`, `mean_rate_aud`, `mean_rate_vis`, `median_rate_aud`, `median_rate_vis`, `obs_diff = mean_aud - mean_vis`, `log2_ratio = log2((mean_aud + 0.1) / (mean_vis + 0.1))`, `modulation_index = (mean_aud - mean_vis) / (mean_aud + mean_vis)` (NaN if both means are 0), `shift_direction` in {`aud_higher`, `vis_higher`, `none`}.
   - **Primary test, `chunk_permutation_test` on detrended rates:** first `R_d = detrend_rates(R, block_of_trial, detrend_window_blocks)`. Within each block, split the selected trials (in order) into contiguous chunks of `chunk_size` trials (merge a final chunk shorter than `chunk_size // 2` into the previous one). Each chunk inherits its block's context label. For each permutation, permute the chunk-level label vector (this preserves the aud/vis chunk counts), expand to trial level, and compute `mean_aud - mean_vis` for every unit at once on `R_d` with a matrix product (`R_d @ w`, `w` = `+1/n_aud` or `-1/n_vis` per trial). `n_perm` permutations with `rng = np.random.default_rng(seed)`; `p_perm = (1 + sum(|perm_diff| >= |obs_diff_detrended|)) / (n_perm + 1)`. Store `obs_diff_detrended`, `p_perm`, `n_chunks_aud`, `n_chunks_vis`. All descriptive columns below stay on the raw (undetrended) rates.
   - Descriptive only: `p_mwu` from `scipy.stats.mannwhitneyu(aud_rates, vis_rates, alternative="two-sided")`; set `p_mwu = 1.0` only if either group is empty or all pooled values are identical (a unit silent in one context but active in the other is a genuine effect and must keep its real p).
   - Block-level metrics: `block_means` (mean rate per block, stored as a list column or 6 columns `block_mean_0..5`); `block_consistency` = fraction of the 5 adjacent block pairs (b, b+1) whose sign of (aud block mean - vis block mean) matches `sign(obs_diff)`, where a tie (sign 0) counts as a mismatch; NaN if `obs_diff == 0`. `full_block_separation` = True if every aud block mean is strictly greater than every vis block mean or vice versa.
   - After concatenating all sessions, `q_bh = scipy.stats.false_discovery_control(p_perm, method="bh")` across ALL units (report the total count).
   - `is_significant = q_bh < alpha`; `is_robust = is_significant & (block_consistency >= min_consistency) & (max(mean_aud, mean_vis) >= min_rate)`.
   Join unit metadata from `units.parquet`: `session_id`, `unit_index`, `unit_id`, `structure`, `location`, `ccf_ap`, `ccf_dv`, `ccf_ml`, `task_firing_rate`, `is_not_drift`, `presence_ratio`. Save `data/precomputed/context_baseline_shifts.parquet` sorted by `session_id`, `unit_index`.

4. **Region summary** (`region_summary`). Group by `structure` across sessions and by (`session_id`, `structure`): `n_units`, `n_significant`, `n_robust`, `frac_significant`, `frac_robust`, `n_aud_higher_robust`, `n_vis_higher_robust`, `median_abs_log2_ratio_robust` (NaN if none), `n_sessions` (across-session table only). Save `data/precomputed/context_baseline_shifts_by_region.parquet` (sorted by `frac_robust` desc) and `data/precomputed/context_baseline_shifts_by_region_session.parquet`.

5. **Run**: `/opt/miniconda3/bin/python3 code/analyze_context_baseline_shift.py`. Print total units, n significant, n robust, fraction with `full_block_separation`, top 15 structures by `frac_robust` with n >= 20, per-session counts, runtime (expect well under 5 minutes: 2000 permutations x a (2000 x 480) @ (480,) product per session). Write `data/precomputed/context_baseline_shifts_summary.json` with all parameters (including `chunk_size` and `detrend_window_blocks`), all counts, and a `flag_semantics` field stating that `is_robust` is the default for tab 1 dots and `is_significant` is the liberal flag.

6. **Sanity checks** in `code/check_context_baseline_shift.py` (PASS/FAIL per check, non-zero exit on failure):
   (a) for 3 randomly chosen robust units (seed 0), recompute `mean_rate_aud`/`mean_rate_vis` directly from `aligned_spikes.parquet` and `trials.parquet` with an independent code path (boolean masks, no groupby) and assert equality within 1e-6;
   (b) calibration on session 759434_2025-02-04, using the session-wide detrended matrix `R_d` and `n_perm=500`, seed 0:
       (i) **pure null**: for each of the six same-context block pairs (0,2), (2,4), (0,4), (1,3), (3,5), (1,5), label one block A and the other B, restrict the chunk test to the trials of those two blocks, and compute the fraction of units with `p_perm < 0.05`; assert the mean over the six pairs is <= 0.10 and the max is <= 0.15;
       (ii) **rank check**: enumerate all `itertools.combinations(range(6), 3)` assignments of 3 aud blocks (20; drop the complement of the truth so 19 remain including the truth), rerun the chunk test under each labelling, and assert the real labelling ranks 1st by fraction of units with `p_perm < 0.05`; print the shuffle mean and max with a note that shuffled labellings agree with the truth on 2 or 4 of 6 blocks and are therefore only partially null;
       (iii) for documentation, report the same real-vs-shuffle fractions for `p_mwu` (expected to show the anticonservativeness) and for the undetrended chunk test;
   (c) assert `q_bh >= p_perm` for all rows, `is_robust` implies `is_significant`, and `p_perm >= 1/(n_perm+1)`;
   (d) detrend sanity: for one unit, assert the detrended rate vector has mean within 1e-6 of 0 over the interior (trials at least half a window from the edges) and that `detrend_rates(R, ..., 0)` returns `R` unchanged.

7. **Visual check**: write `work/context-baseline-shift-analysis/data/example_units.png` with matplotlib (already installed; add `matplotlib` to `requirements.txt`): for the 4 most robust units (lowest `p_perm`, then largest `|log2_ratio|`), per-trial baseline rate vs trial index coloured by context, block boundaries as vertical lines, title with structure and p-values. For the Results narrative only.

8. **Record artifacts.** Copy `context_baseline_shifts_summary.json`, the by-region table exported as `by_region.csv` (top 40 rows), and the sanity-check stdout as `checks.txt` into `work/context-baseline-shift-analysis/data/`. Update `code/README.md` with the script, options, output schema, the detrend + chunk permutation design (one paragraph), and the flag semantics (`is_robust` default for tab 1, `is_significant` liberal).

Expected artifacts: `code/analyze_context_baseline_shift.py`, `code/check_context_baseline_shift.py`, `data/precomputed/context_baseline_shifts.parquet`, `data/precomputed/context_baseline_shifts_by_region.parquet`, `data/precomputed/context_baseline_shifts_by_region_session.parquet`, `data/precomputed/context_baseline_shifts_summary.json`, `work/context-baseline-shift-analysis/data/{context_baseline_shifts_summary.json,by_region.csv,checks.txt,example_units.png}`, updated `code/README.md` and `requirements.txt`.

# Results

## Summary
`code/analyze_context_baseline_shift.py` screened all 6,685 QC-pass units across the 5 sessions for context-dependent baseline shifts using the approved design (pre-stimulus window -1.5 to -0.06 s, first 10 trials of each block excluded, two-block moving-average detrend, 20-trial chunk permutation with 2,000 permutations, BH across all units). Runtime was 2.6 s. 1,692 units (25.3%) are significant at q < 0.05 and 1,642 (24.6%) pass the robust criteria (736 aud-higher, 906 vis-higher); 20.6% of all units show full block separation. The calibration script confirms the test is well behaved on session 759434: the pure null flags 3.9% of units on average (max 5.0%), and the real labelling ranks 1st of 19 block labellings (44.5% vs shuffle mean 5.3%), whereas the trial-level Mann-Whitney test flags 50% of units even under shuffled labels. The example figure shows units in CP and MOs that are nearly silent in one context and fire at 5 to 25 Hz baseline in the other.

## Artifacts
- `code/analyze_context_baseline_shift.py` — analysis script (CLI parameters as planned, plus `--detrend-window-blocks`).
- `code/check_context_baseline_shift.py` — sanity checks and null calibration; exits non-zero on failure.
- `code/plot_context_baseline_examples.py` — figure script for the 4 most robust units.
- `data/precomputed/context_baseline_shifts.parquet` — 6,685 rows x 37 columns (per-unit table for tab 1).
- `data/precomputed/context_baseline_shifts_by_region.parquet` — per-structure summary across sessions (62 structures).
- `data/precomputed/context_baseline_shifts_by_region_session.parquet` — per (session, structure) summary.
- `data/precomputed/context_baseline_shifts_summary.json` — parameters, flag semantics, counts.
- `data/context_baseline_shifts_summary.json`, `data/by_region.csv` (top 40 structures), `data/checks.txt`, `data/example_units.png` (under this work folder).
- `code/README.md` updated (design, flags, schema); `requirements.txt` gained `matplotlib`.

## Key numbers

| | count | fraction |
|---|---|---|
| units screened | 6,685 | |
| significant (q_bh < 0.05) | 1,692 | 25.3% |
| robust (default tab-1 flag) | 1,642 | 24.6% |
| robust aud-higher / vis-higher | 736 / 906 | |
| full block separation | | 20.6% |

Per session robust counts: 664851 232/1118, 668755 407/1184, 713655 361/1666, 742903 388/1973, 759434 254/744.

Top structures by fraction robust (n >= 20 units): OLF 0.40, AON 0.39, ACAd 0.37, CP 0.35, ORBl 0.35, TEa 0.33, FRP 0.33, SSp 0.32, VISal 0.31, MOp 0.30, VISam 0.30, DP 0.29, MOs 0.26 (298 of 1150), SSs 0.25, PL 0.24. Directions are mixed within most regions (e.g. CP 77 aud-higher vs 85 vis-higher; ACAd 18 vs 40).

Calibration (session 759434, n_perm 500): pure null mean 0.039 / max 0.050 per same-context block pair; real labelling 0.445 vs 18 partially-null shuffles mean 0.053 / max 0.147, rank 1/19. For documentation: trial-level MWU real 0.610 vs shuffles mean 0.503 / max 0.628; undetrended chunk test real 0.397 vs shuffles mean 0.229 / max 0.430.

## Step-by-step
1. Script written with the planned functions (`per_trial_baseline_rates`, `detrend_rates`, `chunk_permutation_test`, `unit_stats`, `region_summary`, plus `select_trials`, `rate_matrix`, `add_flags`, `make_chunks`, `contrast_weights`). Docstring states design and flag semantics.
2. Per-trial rates computed on the full unit x selected-trial grid (455 to 485 selected trials per session).
3. Per-unit statistics as specified. `p_mwu` is set to 1.0 only when a group is empty or all pooled values are identical. `block_consistency` treats ties as mismatches and is NaN when `obs_diff == 0`; non-alternating adjacent block pairs are skipped (not needed for the real data, where all blocks alternate).
4. Region summaries written. First run crashed with a `KeyError` in the per-session grouping because pandas excludes the grouping column from the group; fixed by falling back to `n_sessions = 1` and rerun.
5. Full run: 2.6 s; summary JSON includes `chunk_size`, `detrend_window_blocks` and `flag_semantics`.
6. Checks: 8/8 PASS. Per the reviewer's note, check (d) asserts a constant input detrends to zeros and that window 0 is the identity, and only reports (not asserts) the interior mean of one real unit (raw 1.41 Hz vs detrended -0.15 Hz).
7. Figure `example_units.png` written (matplotlib was already installed; added to `requirements.txt`).
8. Artifacts copied; `code/README.md` updated.

Deviations: the figure code lives in its own script `code/plot_context_baseline_examples.py` rather than inside the check script, to keep the check script dependency-free of matplotlib. No other deviations.

# Assessment

## Verdict
accomplished

## Reasoning
The Goal is met in full. `code/analyze_context_baseline_shift.py` implements the approved design (pre-stimulus window -1.5 to -0.06 s, first 10 trials per block excluded, two-block centred moving-average detrend, 20-trial within-block chunk permutation with 2000 permutations, BH across all 6,685 units) with every planned CLI parameter, and writes the per-unit table, both region summaries and the summary JSON. Independent read-only verification of `data/precomputed/context_baseline_shifts.parquet` confirms 6,685 rows x 37 columns matching the schema in `code/README.md` exactly (no missing or extra columns), 1,692 significant / 1,642 robust (736 aud-higher, 906 vis-higher), 20.6% full block separation, per-session counts identical to the Results table, `q_bh` and `is_robust` recomputed from `p_perm`/`block_consistency`/mean rates with zero mismatches, `is_robust` implies `is_significant`, `q_bh >= p_perm`, rows sorted by (session_id, unit_index), one NaN-CCF unit retained, and by-region totals summing to 6,685 units / 1,642 robust. The by-region parquet's top 40 rows match `by_region.csv` and the README's top-structure list. Re-running the script into a scratch directory reproduced all three parquet files bit-for-bit (every column identical, including `p_perm`, `q_bh`, `is_robust`), so the seeded pipeline is reproducible. The example figure shows the four units the stated selection rule actually picks (713655 CP 2163/2162, 668755 MOs 775/908) and they are visually unambiguous context shifts.

Evidence quality is strong. The calibration script passes 8/8 on the planned session (pure null mean 0.039 / max 0.050; real labelling rank 1/19), and on session 668755_2023-08-31, which the executor never tuned against, it also passes 8/8 with n_perm 300: pure-null mean 0.027 / max 0.037, real labelling 0.465 vs partially-null shuffles mean 0.051 / max 0.221, rank 1/19, while the descriptive MWU flags 0.505 of units under shuffled labels (confirming the anticonservativeness argument). The test is therefore well calibrated beyond the session it was developed on. Plan adherence is good: the only deviations are the separate `code/plot_context_baseline_examples.py` (documented, sensible) and the check (d) change to constant-input/identity assertions, which was explicitly requested in the Review Notes. Two non-blocking observations for the tab 1 consumer: 765 units sit at the permutation floor p_perm = 1/2001, so ranking by `p_perm` alone is uninformative among the strongest units (use `|log2_ratio|` or `|obs_diff|` as a tiebreak, as the figure script does); and the region table sorts small structures (n = 1 to 15) to the top, so the viewer should filter on `n_units` when presenting region fractions.

## Recommended next status
done

# Review Notes
Approved: the detrend + 20-trial chunk permutation design was re-verified read-only on 759434_2025-02-04 with the plan's exact window (`round(2 * mean selected trials per block)` = 162): pure-null (same-context block pairs) mean 4.2% / max 5.6% of units at p < 0.05, partially-null shuffles mean 5.3% / max 14.7%, real labelling 44.5% and rank 1 of 19, so the step 6(b) thresholds are comfortably met; all five sessions have 6 alternating blocks with 455 to 485 selected trials, and the (~2000 x 480) @ (480 x 2000) permutation product is sub-second per session. One caveat for the executor: the step 6(d) assertion that the detrended vector has interior mean within 1e-6 of 0 does not hold for a centred moving average (on real data 99.9% of units exceed 1e-6; median |interior mean| 0.02 Hz, max 0.8 Hz), so replace it with (i) a constant-rate input yields all zeros (|R_d| < 1e-9 everywhere, edges included), (ii) `detrend_rates(R, ..., 0)` returns `R` unchanged, and (iii) for the chosen real unit report, without asserting, the interior mean of `R_d` alongside that of `R`.
