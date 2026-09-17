# Goal

Build an interactive web interface, using the Python Shiny package, to visualize neural data from the Allen Institute Dynamic Routing project: raster plots, PSTHs, and unit locations in a 3D mouse brain.

The app has two tabs, each an interactive data exploration viewer with the same layout: a 3D interactive mouse brain on the right with dots marking unit locations, and controls on the left to select a specific unit, brain region, and analysis window, which drive raster plots and PSTHs shown separately for auditory-rewarded and visual-rewarded blocks.

- **Tab 1 (context-dependent baseline shifts):** brain regions and units whose baseline firing rate differs between auditory and visual context blocks. Dots mark units with significant context-dependent baseline shifts.
- **Tab 2 (rule-change units):** brain regions and units that fire specifically around rule changes (block switches). Dots mark rule-updating units. Left panel additionally allows selecting context.

Implementation constraints from the mission:
- Source data lives in `nwb_sessions/` but the data location must be configurable via a single variable.
- All precomputed data is saved in `data/precomputed/`.
- All scripts live in `code/` and must be reproducible.
- The viewer is a Python Shiny app.

# Background

Data: five NWB files in `nwb_sessions/`, one per mouse per session (6 to 9 GB each), from head-fixed mice performing a visual-auditory cross-modal go/no-go attention-switching task with Neuropixels recordings. `nwb_sessions/nwb_explanation.md` documents the key tables: `units` (spike times, waveform features, SpikeInterface QC metrics, `default_qc` flag, CCF coordinates `ccf_ap`/`ccf_dv`/`ccf_ml`, `structure`), `intervals/trials` (per-trial `block_index`, `rewarded_modality` in {aud, vis}, `is_block_switch`, `stim_name`, `stim_start_time`, `quiescent_start_time`/`quiescent_stop_time`, response and reward times, hit/miss/FA/CR flags), `intervals/epochs` (task vs RF mapping vs spontaneous), `intervals/performance`, and `general/electrodes`.

Verified on session 759434_2025-02-04: 2285 units, of which 744 pass `default_qc`; 545 trials across 6 blocks alternating aud/vis rewarded modality; stimuli are catch, sound1, sound2, vis1, vis2; no opto trials in that session. The task epoch is `DynamicRouting1`.

Because the NWB files are large, the app must read only compact precomputed artifacts (aligned spikes, per-unit statistics, brain mesh) rather than opening NWB files at runtime.

Environment: Python 3.13 (miniconda), pynwb 3.1.3, h5py 3.15.1, shiny 1.7.0 available; `uv` available.

# Agent provenance
Who ran what (requested in mission.md):
- Claude Code main session (Claude Fable 5.1) ran the asta-assistant `run` router and its brainstorm / plan-work / do-work / save-work skills for items 1 to 6, wrote all code and docs, ran all scripts and screenshots. Item 7 (rule-change-windows-v2) was executed by Claude Code directly at the scientist's direction; asta-assistant plan review was skipped, work review was kept.
- Claude subagents (fresh-context general-purpose agents spawned by the main session) ran the asta-assistant critic skills review-plan and review-work for every item (7 plan reviews incl. 4 rejections; 8 work assessments incl. 1 partial).
- Asta CLI: `asta auth login`; `asta documents --index-path work/index.yaml add` in each save-work (index at .asta/documents/index.yaml, symlinked from work/index.yaml); `asta autodiscovery run|experiments|experiment` to consult the hosted AutoDiscovery run DR-basic-exploration (experiment node_3_10) on 2026-09-15.
- Asta hosted service: AutoDiscovery run DR-basic-exploration (50 experiments, run 2026-05-27 by the scientist). No other Asta service was used.
- Git: repository initialised 2026-09-15 at the scientist's request; changes staged, no commits made yet.

# Completed Work
- [project-scaffold-and-inventory](work/project-scaffold-and-inventory/README.md) — Create code/ and data/precomputed/ layout, a config module exposing DATA_DIR, and a reproducible script that scans all NWB sessions into a session/unit manifest.
- [precompute-aligned-spikes](work/precompute-aligned-spikes/README.md) — Extract stim-aligned spike times per trial with block labels for QC-passing units into compact per-session files.
- [context-baseline-shift-analysis](work/context-baseline-shift-analysis/README.md) — Per unit, test pre-stimulus baseline rate in aud vs vis blocks; output a table with CCF coords, structure, effect size, and significance for tab 1.
- [ccf-brain-3d-component](work/ccf-brain-3d-component/README.md) — Fetch the Allen CCF root brain mesh into data/precomputed and build a reusable plotly 3D brain figure with overlaid unit dots.
- [rule-change-unit-analysis](work/rule-change-unit-analysis/README.md) — Per unit, compare early post-switch trials vs late-block trials in both switch directions; output a table of rule-updating units for tab 2.
- [shiny-app-tabs](work/shiny-app-tabs/README.md) — Build the Shiny app with both tabs wired to precomputed tables: selectors, 3D brain, raster, and PSTH plots.
- [rule-change-windows-v2](work/rule-change-windows-v2/README.md) — Revise tab 2 to the scientist's phase definitions (instruction 0-4, early 5-14, late = last 10), organise by block context, Kruskal-Wallis + post-hoc trial-shuffle test as primary, and redesign the plots and app layout.

# Pending Work
