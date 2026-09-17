# code/

Reproducible scripts for the Dynamic Routing interactive interface. Run everything
from the repository root with `/opt/miniconda3/bin/python3` (or any environment with
`pip install -r requirements.txt`).

## Configuration (`config.py`)

| Variable | Env override | Default |
|---|---|---|
| `DATA_DIR` | `DR_DATA_DIR` | `<repo>/nwb_sessions` |
| `PRECOMPUTED_DIR` | `DR_PRECOMPUTED_DIR` | `<repo>/data/precomputed` |
| `TASK_SCRIPT_NAME` | — | `DynamicRouting1` (epochs `script_name` of the task epoch) |

Helpers: `list_sessions()` (sorted `*.nwb` in `DATA_DIR`), `session_id_from_path()`.

## Scripts

### `inventory_sessions.py`
Scans every `.nwb` in `DATA_DIR` using h5py (metadata only, never spike times) and writes
`sessions_manifest.json` (detailed, keyed by session id) and `sessions_manifest.csv`
(one row per session) into `PRECOMPUTED_DIR`.

```
python code/inventory_sessions.py [--data-dir DIR] [--out-dir DIR]
```

Output is deterministic apart from the `generated_at` timestamp.

### `precompute_aligned_spikes.py`
Extracts stimulus-aligned spike times for QC-passing units (or all with `--all-units`)
into compact parquet files the app reads instead of the NWB files.

```
python code/precompute_aligned_spikes.py [--data-dir D] [--out-dir D] [--sessions SID ...]
                                          [--window PRE POST] [--all-units] [--overwrite]
```
Defaults: window `-2.0 3.0` s relative to `stim_start_time`, QC-pass units only, skip
sessions whose outputs already exist. Runtime is about 1 s per session.

Outputs per session in `PRECOMPUTED_DIR/<session_id>/`:

| file | rows | key columns |
|---|---|---|
| `units.parquet` | one per unit | `session_id`, `unit_index` (row in NWB units table), `unit_id`, `structure`, `location`, `ccf_ap/dv/ml` (µm), `default_qc`, QC metrics (`firing_rate`, `presence_ratio`, `isi_violations_ratio`, `amplitude_cutoff`, `snr`, `half_width`, `peak_to_valley`, `num_spikes`, `activity_drift`, `is_not_drift`), `electrode_group_name`, `peak_channel`, `n_spikes_task`, `task_firing_rate` (Hz over the DynamicRouting1 epoch), `n_spikes_aligned` |
| `trials.parquet` | one per trial | `trial_index`, `block_index`, `trial_index_in_block`, `rewarded_modality` (aud/vis), `switch_direction` (`aud_to_vis` / `vis_to_aud`, NaN for block 0), `stim_name`, outcome/type flags (`is_hit`, `is_miss`, `is_false_alarm`, `is_correct_reject`, `is_go`, `is_nogo`, `is_catch`, `is_target`, `is_nontarget`, `is_vis_stim`, `is_aud_stim`, `is_response`, `is_rewarded`, `is_opto`, `is_repeat`, `is_instruction`, `is_correct`, `is_block_switch`), `stim_start_time` (absolute s), and stim-relative event times `quiescent_start_rel`, `quiescent_stop_rel`, `stim_stop_rel`, `response_window_start_rel`, `response_window_stop_rel`, `response_time_rel`, `reward_time_rel` |
| `aligned_spikes.parquet` | one per spike in window | `unit_index` (int32), `trial_index` (int16), `t_rel` (float32 s relative to stim onset) |

Cross-session: `PRECOMPUTED_DIR/units_all.parquet` (all sessions' units tables concatenated)
and `PRECOMPUTED_DIR/precompute_manifest.json` (parameters, per-session counts, sizes, runtimes).

### `validate_aligned_spikes.py`
Checks one session's precomputed tables against the raw NWB (brute-force spike recount for
3 units x 20 random trials, window bounds, table sizes, per-unit counts, pre-stimulus rate
sanity). `python code/validate_aligned_spikes.py --session 759434_2025-02-04`; exits non-zero on failure.

### `analyze_context_baseline_shift.py` (tab 1 statistics)
Screens every QC-pass unit for a context-dependent baseline (pre-stimulus) firing-rate
shift between auditory-rewarded and visual-rewarded blocks.

```
python code/analyze_context_baseline_shift.py [--baseline-window -1.5 -0.06] [--exclude-first-n 10]
    [--chunk-size 20] [--detrend-window-blocks 2] [--n-perm 2000] [--alpha 0.05]
    [--min-consistency 0.8] [--min-rate 0.1] [--seed 0] [--sessions SID ...]
```

**Design.** Per-trial baseline rates are autocorrelated within blocks and drift slowly across
the session, so a trial-level Mann-Whitney test (or a chunk permutation on raw rates) is
anticonservative: on session 759434 a pure null (two same-context blocks labelled against each
other) still flagged about 38% of units. The primary test therefore (1) detrends each unit's
rate vector with a centred moving average spanning two blocks of selected trials (~160 trials),
(2) groups trials into contiguous 20-trial chunks within blocks, and (3) permutes chunk context
labels (2000 permutations) to get `p_perm` for `mean_aud - mean_vis`; `q_bh` is
Benjamini-Hochberg across all units. With these defaults the pure null gives ~4% of units at
p < 0.05 and the real labelling ~45%. Mann-Whitney `p_mwu` is kept as a descriptive column only.

**Flags.** `is_robust` (default for tab 1 dots) = `q_bh < alpha` AND `block_consistency >= 0.8`
(sign of the aud-vs-vis difference agrees in at least 4 of 5 adjacent block pairs) AND max
context mean rate >= 0.1 Hz. `is_significant` = `q_bh < alpha` only (liberal).

Outputs in `PRECOMPUTED_DIR`:

| file | rows | key columns |
|---|---|---|
| `context_baseline_shifts.parquet` | one per unit | unit metadata (`session_id`, `unit_index`, `unit_id`, `structure`, `location`, `ccf_*`, `task_firing_rate`, `is_not_drift`, `presence_ratio`), `n_trials_aud/vis`, `mean_rate_aud/vis`, `median_rate_aud/vis`, `obs_diff`, `log2_ratio` (log2((aud+0.1)/(vis+0.1))), `modulation_index`, `shift_direction` (aud_higher/vis_higher/none), `obs_diff_detrended`, `p_perm`, `q_bh`, `n_chunks_aud/vis`, `p_mwu`, `block_mean_0..5`, `block_consistency`, `full_block_separation`, `is_significant`, `is_robust` |
| `context_baseline_shifts_by_region.parquet` | one per structure | `n_units`, `n_significant`, `n_robust`, `frac_significant`, `frac_robust`, `n_aud_higher_robust`, `n_vis_higher_robust`, `median_abs_log2_ratio_robust`, `n_sessions` |
| `context_baseline_shifts_by_region_session.parquet` | one per (session, structure) | same without `n_sessions` |
| `context_baseline_shifts_summary.json` | | parameters, flag semantics, counts, per-session counts, top structures |

`check_context_baseline_shift.py` reruns the null calibration (pure-null and 19-labelling rank
check), an independent recount of context means, flag consistency and detrend sanity checks;
exits non-zero on failure. `plot_context_baseline_examples.py --out FILE` draws the 4 most robust units.

### `fetch_ccf_mesh.py` and `brain3d.py` (3D brain component)
`python code/fetch_ccf_mesh.py` loads the brainglobe `allen_mouse_25um` atlas (cached in
`~/.brainglobe`) and writes `PRECOMPUTED_DIR/ccf_meshes/<acronym>.npz` (`vertices` float32 µm in
atlas axis order ap/dv/ml, `faces` int32) for `root` and every structure present in
`units_all.parquet`, plus `root_decimated.npz` (25% of faces, via fast-simplification) and
`index.json` (id, name, rgb, sizes, bounds per structure).

`brain3d.py` API:
- `load_mesh(acronym, mesh_dir, prefer_decimated=True) -> (vertices, faces)`; `load_mesh_index(mesh_dir)`.
- `to_plot_coords(ap, dv, ml) -> (x, y, z)` with `x = ml`, `y = -ap`, `z = -dv` (right hemisphere at low x, anterior at high y, dorsal up). Applied identically to meshes and unit dots.
- `brain_figure(units_df, color_col=None, hover_cols=(...), selected_unit=(session_id, unit_index) | None, region_meshes=[...], mesh_opacity, region_opacity, colorscale, color_range, marker_size, title, colorbar_title) -> plotly Figure` — translucent root mesh, optional coloured region meshes, unit `Scatter3d` with `customdata = [session_id, unit_index, ...hover cols]`, highlighted selected unit, `uirevision="brain"` so the camera persists across updates; units with NaN coordinates are dropped and counted in an annotation.
- `unit_click_to_key(click_data) -> (session_id, unit_index) | None` for click-to-select in the app.

`check_brain3d.py` verifies mesh export and unit/mesh alignment (bbox and annotation-volume
containment >= 0.95 for seven structures). `brain3d_demo.py --out FILE.html` writes a standalone demo.

### `analyze_rule_change_units.py` (tab 2 statistics)
Screens every QC-pass unit for rule-change responses using the scientist's phase definitions
(2026-09-15): within every block the first 5 trials are **instruction** trials (`trial_index_in_block`
0 to 4), **early** is trials 5 to 14, **late** is the last 10 trials of the block. A block's
**context** is its rewarded modality (aud / vis); results are given per context and pooled
(`context` = aud, vis, both) over the 5 switch blocks (block 0 excluded), in a baseline window
(-1.5 to -0.06 s) and a stimulus-evoked window (0.0 to 0.5 s).

```
python code/analyze_rule_change_units.py [--instruction-n 5] [--early-window 5 15] [--late-n 10]
    [--n-perm-kw 5000] [--chunk-size 5] [--n-perm 20000] [--alpha 0.05] [--p-candidate 0.01] [--p-liberal 0.05]
    [--min-consistency 0.8] [--min-rate 0.1] [--profile-start-n 50] [--profile-end-n 20] [--seed 0]
```

**Definition (scientist, 2026-09-15): a rule-updating unit's mean rate in the instruction trials is
greater than in the early trials AND greater than in the late trials** (`<w>_kw_is_instruction_higher`
per window, `kw_is_instruction_higher` = either window, `instruction_higher_window`). This is the
default set shown in the viewer; 3,532 of 6,685 units (52.8%) meet it in at least one window, close
to the roughly 1/3-per-window chance level, so the statistical tiers below are offered as optional
stricter filters.

**Statistical tiers.** Per unit, a Kruskal-Wallis H across instruction / early / late trial rates, with a null distribution
from shuffling the phase labels across those trials within each block (5,000 shuffles); post-hoc
one-sided Dunn-style contrasts (mean rank of instruction trials minus early, minus late) use the same
shuffles. `<w>_kw_is_rule_updating` = omnibus `kw_p_perm` < 0.05 AND both post-hoc p < 0.05 AND
rate >= 0.1 Hz (841 units pooled); `<w>_kw_is_candidate` uses 0.01 (472). **BH correction is on hold**: `<w>_kw_q_bh`
(across units) and `<w>_kw_q_unit_early/late` (within the unit's 12-test family of contexts x
windows x contrasts) are supplementary columns, and `<w>_kw_is_bh` is the BH-corrected variant.
Calibration with mid-block pseudo phases (instruction := trials 40 to 44, early := 45 to 54,
late := 20 to 29) flags 1.2% of units on average (omnibus alone 12.7%, because trial shuffles ignore
autocorrelation; the two post-hoc requirements remove that). For the evoked window,
`ev_kw_p_perm_residualized` repeats the test after removing stimulus-identity x context means
(instruction trials are all rewarded-modality targets).

**Secondary screens** (detrended (block, phase)-stratified 5-trial chunk permutation, 20,000
permutations): early vs late (`bl_/ev_` prefix) and instruction vs late (`bl_inst_/ev_inst_`), each
with `is_robust` (BH), `is_candidate` (p < 0.01), `is_liberal` (p < 0.05) tiers gated by block
consistency >= 0.8 (with 2 or 3 blocks per context this means all blocks agree) and rate >= 0.1 Hz.

Outputs in `PRECOMPUTED_DIR`:

| file | rows | key columns |
|---|---|---|
| `rule_change_units.parquet` | 3 per unit (`context`) | unit metadata, `direction`, `n_blocks`, `n_trials_instruction/early/late`; per window `bl_/ev_`: `mean_instruction`, `mean_early`, `mean_late`, `mean_rest`, `kw_h`, `kw_p_perm`, `kw_p_chi2`, `kw_d_early`, `kw_p_early`, `kw_d_late`, `kw_p_late`, `kw_log2_ratio`, `kw_instruction_higher`, `kw_direction_sign`, `kw_q_bh`, `kw_q_unit_early/late`, `kw_is_rule_updating/candidate/bh`; secondary contrasts `diff`, `log2_ratio`, `modulation_index`, `direction_sign`, `block_diff_<b>`, `consistency`, `p_perm`, `q_bh`, tiers (and the same with `inst_`); unit-level `kw_is_rule_updating`, `kw_is_candidate`, `kw_is_bh`, `kw_window`, `is_rule_updating`, `is_candidate`, `is_liberal`, `inst_*`, `rule_updating_window`, `candidate_window` |
| `rule_change_profiles.parquet` | unit x context x align x window x position | `align` = `block_start` (`rel_trial` 0..49) or `block_end` (-20..-1), `mean_rate`, `n_blocks` |
| `rule_change_profiles_by_region.parquet` | structure x context x align x window x position | population mean/sem over KW rule-updating units (structures with >= 20 units) |
| `rule_change_by_region.parquet`, `rule_change_by_region_session.parquet` | per structure (/ session) | `n_units`, `n_instruction_higher`, `frac_instruction_higher`, `n_kw_rule_updating`, `frac_kw_rule_updating`, `n_kw_candidate`, `n_kw_bh`, secondary counts, `n_early_higher`, `n_late_higher`, `n_sessions` |
| `rule_change_summary.json` | | parameters, flag semantics, tier counts, `calibration` (written by the check script) |

`check_rule_change_units.py` runs recounts, the pseudo-window calibrations for both the chunk
tests and the KW flag, and flag/profile invariants; `plot_rule_change_examples.py --out FILE`
draws the 4 rule-updating units (definition) with the smallest KW permutation p.

### App (`app.py`, `app_data.py`, `app_plots.py`)
`shiny run code/app.py --port 8000`. `app_data.py` is the cached data layer (per-session parquet
tables, unit spike slices, statistics tables, profiles, summaries); `app_plots.py` holds the
matplotlib helpers (`raster_psth`, `per_trial_rate_plot` for tab 1; `block_timeline_plot`,
`context_raster_psth`, `aligned_means_plot` for tab 2); `app.py` wires the two tabs. Each tab is a
`layout_sidebar`: controls in the sidebar, a plots column (7/12) and a brain column (5/12) with the
selected-unit statistics and the region table under the brain. Tab 2 selectors: context (aud / vis /
both), test (KW instruction > early and late, or the two chunk-permutation contrasts), tier, and
statistics window; dots, unit list and region table all follow the selected window. Tab 2 plots:
whole-session block timeline; one raster with all blocks stacked in session order (plain spike
marks, shaded bands for instruction / early / late trials, rewarded context labelled per block) with an instruction / early / late PSTH; block-start
and block-end aligned means per context; an instruction-trial dynamics panel (`instruction_dynamics_plot`: the selected unit's mean rate at
block positions 0 to 14 per context, and the displayed population's fold change relative to each unit's
late-trial rate with SEM, plus a step-vs-exponential-decay AIC comparison over the five instruction trials,
replicating AutoDiscovery followup-III experiments node_4_16 / node_4_45); and, for the selected unit, the permutation null recomputed
live for the current analysis window and context (`kw_null_for_unit`, 5,000 shuffles) with the
observed statistics. The region table lists units recorded (all sorted units, from
`sessions_manifest.json`), QC-pass units, rule-updating units for the current selection, the number
expected under the label-shuffle null (sum of per-unit `kw_null_rate_*`), and mean instruction /
early / late rates over the rule-updating units. The 3D brains are plotly `FigureWidget`s served through shinywidgets; clicking a unit
dot sets the session, region and unit selectors (the click handler stores a pending selection that
the selector-update effects consume). `DR_APP_DEFAULT_TAB=tab_rule` opens the app on tab 2 (used by
`check_app.py`). Keep `shinywidgets` imports confined to `app.py`: once imported, plotly
`FigureWidget`s can only be built inside a Shiny session.

`check_app_plots.py [OUT_DIR]` renders the plot helpers for one context unit and one rule-updating
unit; `check_app.py` starts the server, confirms HTTP 200, drives headless Chrome over the DevTools
protocol (software WebGL, 25 s real wait) to screenshot both tabs and probe the DOM (populated
selectors, plots, plotly widget, no output errors).
