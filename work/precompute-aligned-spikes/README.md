---
slug: precompute-aligned-spikes
status: done
plan_review_attempts: 1
---

# Goal
Produce the compact spike data the Shiny app will read instead of the multi-GB NWB files. For each session in `DATA_DIR`, for every unit passing `default_qc` (with an option to include all units), extract spike times aligned to `stim_start_time` for every trial in the DynamicRouting task epoch, within a generous window (e.g. -2.0 s to +3.0 s) so any downstream analysis window fits. Store, per session, a units table (unit_id, session, structure, ccf_ap/dv/ml, QC metrics, firing_rate) and an aligned-spikes table (unit_id, trial_index, relative spike time) in parquet, plus a trials table (trial_index, block_index, rewarded_modality, is_block_switch, trial_index_in_block, stim_name, is_go/is_hit/etc, quiescent and response window times relative to stim onset). Write to `data/precomputed/<session_id>/`.

The script `code/precompute_aligned_spikes.py` must be reproducible, use `config.DATA_DIR`, and be memory-conscious (read spike_times via the ragged index per unit rather than loading the whole column at once if needed). Report total runtime and output sizes.

# Instructions

Paths are relative to the repo root `/Users/pavir/Code/dr-interactive-interface`. Interpreter: `/opt/miniconda3/bin/python3` (pandas 2.3, pyarrow 23, h5py 3.15 installed by [project-scaffold-and-inventory](../project-scaffold-and-inventory/README.md)). Use `code/config.py` for `DATA_DIR`, `PRECOMPUTED_DIR`, `TASK_SCRIPT_NAME`, `list_sessions()`, `session_id_from_path()`.

Verified data facts to rely on: `units/spike_times` is one contiguous float64 array (e.g. 34.5M spikes for 759434) with `units/spike_times_index` giving the cumulative end offset per unit; per-unit spike times are sorted. All trials lie within the DynamicRouting1 epoch. Relative to `stim_start_time`: `quiescent_start_time` is about -1.53 s, `quiescent_stop_time` about -0.06 s, `stim_stop_time` about +0.5 s, `response_window` about +0.06 to +1.0 s, trial `stop_time` about +4.0 s; the inter-trial interval is at least 5.5 s, so a -2.0 to +3.0 s window never overlaps a neighbouring stimulus. `rewarded_modality` and `stim_name` are byte/object strings; boolean columns are dtype bool.

1. **Write `code/precompute_aligned_spikes.py`** with argparse options: `--data-dir` (default `config.DATA_DIR`), `--out-dir` (default `config.PRECOMPUTED_DIR`), `--sessions SID [SID ...]` (subset by session id, default all), `--window PRE POST` (default `-2.0 3.0`, seconds relative to stim onset), `--all-units` (include units failing `default_qc`; default QC-pass only), `--overwrite` (else skip sessions whose outputs already exist). Structure the script as functions `load_trials(f) -> DataFrame`, `load_units(f, session_id, qc_only) -> DataFrame`, `align_spikes(f, units_df, trials_df, window) -> DataFrame`, `process_session(path, ...)`, `main()`.

2. **Trials table** (`load_trials`). From `intervals/trials` read and write one row per trial with columns: `trial_index` (0-based row order; also keep the file's `trial_index` column as `trial_index_nwb` if it differs), `block_index`, `trial_index_in_block`, `rewarded_modality` (str: aud/vis), `is_block_switch`, `stim_name`, `is_vis_stim`, `is_aud_stim`, `is_target`, `is_nontarget`, `is_go`, `is_nogo`, `is_catch`, `is_hit`, `is_miss`, `is_false_alarm`, `is_correct_reject`, `is_response`, `is_rewarded`, `is_opto`, `is_repeat`, `is_instruction`, `is_correct`, `stim_start_time` (absolute s), and relative-to-stim-onset columns `quiescent_start_rel`, `quiescent_stop_rel`, `stim_stop_rel`, `response_window_start_rel`, `response_window_stop_rel`, `response_time_rel`, `reward_time_rel` (NaN where absent). Add `switch_direction` per block: for block b>0 the string `"<prev>_to_<cur>"` (e.g. `aud_to_vis`), NaN for block 0. Decode bytes to str. Save as `<out-dir>/<session_id>/trials.parquet`.

3. **Units table** (`load_units`). From `units` read, for every unit (then filter `default_qc == True` unless `--all-units`): `unit_index` (row number in the NWB units table), `unit_id` (str), `structure`, `location`, `ccf_ap`, `ccf_dv`, `ccf_ml`, `default_qc`, `firing_rate`, `presence_ratio`, `isi_violations_ratio`, `amplitude_cutoff`, `snr`, `half_width`, `peak_to_valley`, `num_spikes`, `electrode_group_name`, `peak_channel`, `activity_drift`, `is_not_drift`. Prepend `session_id`. After alignment (step 4) add `n_spikes_task` (spikes within the task epoch bounds from `intervals/epochs` where `script_name == config.TASK_SCRIPT_NAME`) and `task_firing_rate` = `n_spikes_task / epoch_duration`, and `n_spikes_aligned` (rows in the aligned table for this unit). Save as `<out-dir>/<session_id>/units.parquet`.

4. **Aligned spikes** (`align_spikes`). For each selected unit read only its slice `spike_times[idx[i-1]:idx[i]]` (with `idx = spike_times_index[:]`, start 0 for i=0). For each trial compute `lo = np.searchsorted(spk, stim_start + PRE)` and `hi = np.searchsorted(spk, stim_start + POST)`; collect `spk[lo:hi] - stim_start` as float32, with `unit_index` int32 and `trial_index` int16 repeated. Do this vectorised over trials (searchsorted with the full arrays of `stim_start+PRE` and `stim_start+POST`, then `np.repeat`/`np.concatenate`) so it is fast. Accumulate per-unit arrays in lists and concatenate once per session; build a DataFrame with columns `unit_index`, `trial_index`, `t_rel` and save as `<out-dir>/<session_id>/aligned_spikes.parquet` (pyarrow, `compression="zstd"`, sorted by unit_index then trial_index). Print per-session unit count, row count, runtime, file size. Expected scale: roughly 500 to 2000 QC units per session, 515 to 545 trials, so tens of millions of rows per session; keep everything in numpy until the final DataFrame so memory stays below a few GB.

5. **Cross-session outputs.** After all sessions, concatenate every session's `units.parquet` into `<out-dir>/units_all.parquet`, and write `<out-dir>/precompute_manifest.json` recording the parameters (window, qc_only), per-session counts (n_units, n_trials, n_aligned_rows), output file sizes, runtimes, script version/date, and `data_dir`.

6. **Run on one session first** as a smoke test: `/opt/miniconda3/bin/python3 code/precompute_aligned_spikes.py --sessions 759434_2025-02-04 --overwrite`. Expect 744 units and 545 trials.

7. **Validate** with a short script `code/validate_aligned_spikes.py` (also reusable later): for session 759434_2025-02-04 pick 3 units (first, median-index, last in the units table) and (a) recount spikes in the window directly from the NWB with a brute-force boolean mask for 20 random trials and assert equality with the parquet rows; (b) assert `t_rel.min() >= PRE` and `t_rel.max() < POST`; (c) assert `trials.parquet` has 545 rows and `units.parquet` 744 rows with no NaN `unit_id`; (d) recompute the mean pre-stimulus rate (`-1.5` to `-0.06` s) for one unit from the aligned table and check it is within a plausible factor of the NWB `firing_rate` (report both numbers; this is a sanity check, not an equality assertion). Print PASS/FAIL per check and exit non-zero on failure.

8. **Run all sessions**: `/opt/miniconda3/bin/python3 code/precompute_aligned_spikes.py --overwrite`. Record total runtime and total output size (`du -sh data/precomputed`). Then re-run without `--overwrite` to confirm it skips existing sessions (idempotency).

9. **Record artifacts.** Copy `data/precomputed/precompute_manifest.json` and the validation script's stdout (saved as `validation_759434.txt`) into `work/precompute-aligned-spikes/data/`. Update `code/README.md` with a section for the new script (options, outputs, schema of the three parquet files).

Expected artifacts: `code/precompute_aligned_spikes.py`, `code/validate_aligned_spikes.py`, `data/precomputed/<session_id>/{units,trials,aligned_spikes}.parquet` for all 5 sessions, `data/precomputed/units_all.parquet`, `data/precomputed/precompute_manifest.json`, `work/precompute-aligned-spikes/data/{precompute_manifest.json,validation_759434.txt}`, updated `code/README.md`.

# Results

## Summary
`code/precompute_aligned_spikes.py` extracts stimulus-aligned spikes (window -2.0 to +3.0 s) for all QC-passing units in all 5 sessions into per-session parquet files (`units`, `trials`, `aligned_spikes`), plus a cross-session `units_all.parquet` (6,685 units) and `precompute_manifest.json`. The full run took 4.9 s wall time (about 1 s per session) and produced 242 MB total; the largest session (742903, 1,973 units) yields 14.4 M aligned spikes in a 63.5 MB zstd parquet. A validation script confirmed the tables against the raw NWB for session 759434_2025-02-04 (7/7 checks passed, including a brute-force spike recount). Re-running without `--overwrite` skips all sessions.

## Artifacts
- `code/precompute_aligned_spikes.py` — extraction script (argparse: `--data-dir`, `--out-dir`, `--sessions`, `--window`, `--all-units`, `--overwrite`).
- `code/validate_aligned_spikes.py` — validation against the raw NWB, exits non-zero on failure.
- `data/precomputed/<session_id>/{units,trials,aligned_spikes}.parquet` — for all 5 sessions.
- `data/precomputed/units_all.parquet` — 6,685 QC-pass units across sessions with CCF coords and structure.
- `data/precomputed/precompute_manifest.json` — params, per-session counts, sizes, runtimes.
- `data/precompute_manifest.json` (under this work folder) — copy of the manifest.
- `data/validation_759434.txt` (under this work folder) — validation stdout.
- `code/README.md` — updated with schema and usage for both scripts.

## Per-session output

| session | units (QC) | trials | aligned spikes | runtime | aligned_spikes.parquet |
|---|---|---|---|---|---|
| 664851_2023-11-15 | 1118 | 534 | 9,773,273 | 0.9 s | 43.0 MB |
| 668755_2023-08-31 | 1184 | 524 | 14,097,836 | 1.1 s | 61.4 MB |
| 713655_2024-08-09 | 1666 | 515 | 12,177,590 | 1.0 s | 53.7 MB |
| 742903_2024-10-22 | 1973 | 538 | 14,384,894 | 1.3 s | 63.5 MB |
| 759434_2025-02-04 | 744 | 545 | 6,843,539 | 0.6 s | 30.1 MB |

## Step-by-step
1. Wrote the script with the planned functions (`load_trials`, `load_units`, `task_epoch`, `align_spikes`, `process_session`, `main`) and CLI options.
2. Trials table as specified, including `switch_direction` per block and stim-relative event columns (float32). The NWB `trial_index` equals the row order in all sessions, so no `trial_index_nwb` column was added.
3. Units table with all listed columns plus `n_spikes_task`, `task_firing_rate`, `n_spikes_aligned`.
4. Alignment via per-unit contiguous slice reads and vectorised `searchsorted` over all trials; single concatenation per session; zstd parquet sorted by unit then trial.
5. `units_all.parquet` and `precompute_manifest.json` written; the manifest merges per-session entries across runs when parameters match, so a no-op re-run keeps all 5 sessions.
6. Smoke test on 759434_2025-02-04: 744 units, 545 trials, 6,843,539 spikes, 0.6 s.
7. Validation: all 7 checks PASS (brute-force recount for units 1, 1209, 2279 across 20 random trials; t_rel in [-2.0, 3.0]; 545 trials; 744 units; no NaN ids; per-unit counts consistent; unit 1209 pre-stim rate 0.21 Hz vs 0.24 Hz task rate). Per the reviewer's note, check (b) uses `<= POST` because of float32 rounding at the upper edge.
8. Full run: 5/5 sessions ok in 4.9 s; `du -sh data/precomputed` = 242 MB. Re-run without `--overwrite` skipped all sessions.
9. Copied the manifest and validation output into `work/precompute-aligned-spikes/data/`; updated `code/README.md`.

Deviations: none. Runtime was far below the plan's "minutes" estimate because `spike_times` is stored contiguously and reads are effectively memory-bandwidth bound.

# Assessment

## Verdict
accomplished

## Reasoning
The Goal is met and every claim in # Results was independently confirmed. All five sessions have `units/trials/aligned_spikes.parquet` under `data/precomputed/<session_id>/`, plus `units_all.parquet` (6,685 rows, all `default_qc == True`, no NaN `unit_id`, one unit with NaN CCF coordinates) and `precompute_manifest.json`; per-session unit/trial/aligned-row counts in the parquet files match the manifest and the README table exactly (e.g. 744 / 545 / 6,843,539 for 759434_2025-02-04). Every column listed in `code/README.md` exists in the units and trials tables with no extras; aligned dtypes are int32/int16/float32 and rows are sorted by (unit_index, trial_index); `t_rel` lies in [-2.0, 3.0] (max hits exactly 3.0 in two sessions only through float32 rounding, as the reviewer anticipated); `switch_direction` takes values `aud_to_vis` / `vis_to_aud` / NaN, is NaN for block 0 and agrees with the block-wise `rewarded_modality` sequence; stim-relative event medians (quiescent -1.53 to -0.06 s, stim_stop +0.50 s, response window +0.06 to +0.99 s) match the documented data facts. `code/validate_aligned_spikes.py` passes 7/7 on two sessions the executor did not validate (664851_2023-11-15 and 742903_2024-10-22 with a different seed), including the brute-force NWB spike recount. Re-running the extractor into a scratch directory for 759434_2025-02-04 reproduces all three parquet tables bit-for-bit (`DataFrame.equals` True) in under 1 s, confirming reproducibility and that the script honours `--out-dir` and `config.DATA_DIR`.

Plan adherence is good: the script has the prescribed functions and CLI flags, uses per-unit contiguous slice reads with vectorised `searchsorted`, writes zstd parquet, and the one documented departure (no `trial_index_nwb` column because `trial_index == arange` in every session) is exactly what the plan allowed. Two minor observations for downstream work, neither affecting the verdict: (1) the committed manifest's `total_runtime_s_this_run` is 0.0 because the final idempotency re-run (no `--overwrite`) rewrote it, so the 4.9 s full-run figure survives only in this README; (2) because `t_rel` is float32, a downstream filter of `t_rel < 3.0` may drop a handful of edge spikes; this is harmless for the app's analysis windows, which are well inside the range.

## Recommended next status
done

# Review Notes
Approved: every step names a concrete file, function, CLI flag, or HDF5 path; all 22 `units/*` and 33 `intervals/{trials,epochs}/*` columns referenced were verified read-only against 759434_2025-02-04.nwb, as were the stated data facts (contiguous float64 `spike_times` of 34,496,042 samples with a uint32 cumulative `spike_times_index`, sorted per-unit spikes, ITI >= 5.54 s, quiescent/stim_stop/response-window offsets, all 545 trials inside the DynamicRouting1 epoch, `trial_index == arange`, 744 QC units), so the h5py per-unit-slice plus vectorised searchsorted approach is realistic in minutes and well under a few GB even for the 1973-QC-unit session. One caveat for the executor: casting `t_rel` to float32 can round values just below POST up to exactly 3.0, so in step 7(b) either apply the window test after the float32 cast or assert `t_rel.max() <= POST` to avoid a spurious FAIL.
