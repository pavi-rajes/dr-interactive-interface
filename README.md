# Dynamic Routing interactive neural viewer

An interactive Python Shiny app for exploring Neuropixels recordings from the Allen Institute
Dynamic Routing task (mice switching between auditory- and visual-rewarded blocks). Two tabs:

1. **Context baseline shifts** — units whose pre-stimulus (baseline) firing rate differs between
   auditory and visual context blocks. Left: pick session, brain region, unit and analysis window;
   see rasters and PSTHs for auditory vs visual blocks and the per-trial baseline rate across the
   session. Right: 3D Allen CCF brain with one dot per unit (colour = log2 aud/vis baseline rate);
   click a dot to select that unit.
2. **Rule-change units** — units that fire more during the instruction trials that follow a block
   switch (rule change) than during the rest of the block. Sidebar: block context (auditory /
   visual / both), test, tier and statistics window. Plots: whole-session block timeline with
   instruction / early / late trials marked; one raster with all blocks stacked in session order (rewarded
   context labelled per block, shaded bands for instruction / early / late trials) with an instruction / early / late PSTH; and
   block-start / block-end aligned mean rates. Right: 3D brain with rule-updating units, the selected
   unit's permutation null (recomputed live for the chosen analysis window) and stored statistics, and a
   region table (units recorded, QC-pass, rule-updating, expected under the null, phase firing rates).

## Run

```bash
/opt/miniconda3/bin/python3 -m pip install -r requirements.txt
shiny run code/app.py --port 8000        # then open http://127.0.0.1:8000
```

Data locations are variables in `code/config.py`, overridable with environment variables:
`DR_DATA_DIR` (raw `*.nwb` sessions, default `nwb_sessions/`) and `DR_PRECOMPUTED_DIR`
(default `data/precomputed/`). The app reads only precomputed artifacts, never NWB files.

## Reproduce the precomputed data

Run from the repository root, in order (total well under a minute after the NWB files are present):

```bash
python code/inventory_sessions.py                # session/unit manifest
python code/precompute_aligned_spikes.py         # stim-aligned spikes, units, trials per session
python code/analyze_context_baseline_shift.py    # tab 1 statistics
python code/analyze_rule_change_units.py         # tab 2 statistics and switch-aligned profiles
python code/fetch_ccf_mesh.py                    # Allen CCF meshes for the 3D brain
```

Checks: `validate_aligned_spikes.py`, `check_context_baseline_shift.py`, `check_rule_change_units.py`,
`check_brain3d.py`, `check_app_plots.py`, `check_app.py` (launches the app and screenshots both tabs).

## Flags shown as dots

- Tab 1: `is_robust` = BH-corrected q < 0.05 on a detrended 20-trial chunk permutation test of
  auditory vs visual baseline rate, sign consistent in >= 4 of 5 adjacent block pairs, and mean
  baseline rate >= 0.1 Hz in at least one context. Untick "Only units with robust context shifts"
  to see every QC-pass unit.
- Tab 2: `kw_is_instruction_higher` (the definition of a rule-updating unit): mean rate in the 5
  instruction trials of a switch block (trials 0 to 4) is greater than in the early trials (5 to 14) and
  greater than in the late trials (last 10), in the baseline or evoked window. Every unit meeting this is
  in the default set (3,532 of 6,685). Optional stricter tiers add a Kruskal-Wallis omnibus with a
  within-block trial-shuffle null plus one-sided post-hoc instruction > early and instruction > late
  (p < 0.05: 841 units; p < 0.01: 472). BH correction is on hold (q-values shown as supplementary).
  The selector also offers the early-vs-late and instruction-vs-late chunk-permutation screens.

See `code/README.md` for the full pipeline, statistical design, and output schemas, and
`nwb_sessions/nwb_explanation.md` for the dataset.

## Code Ocean

Clone this repository into a capsule (`/code`). The raw NWB files are not in git.

- **Reproducible run** (`code/run`, the capsule entrypoint): attach the Dynamic Routing NWB data
  assets; the script finds the first folder under `/data` containing `.nwb` files (or set `DR_DATA_DIR`)
  and writes every precomputed artifact plus the check outputs to `/results/precomputed`. Register
  `/results` as a data asset afterwards. Runtime is a few minutes.
- **Viewer**: attach the precomputed data asset (or upload the local `data/precomputed/` folder, about
  250 MB including the per-session `aligned_spikes.parquet` files) and start the app in a cloud
  workstation with `code/run_app.sh 8080`. The script finds `sessions_manifest.json` under `/data` (or
  set `DR_PRECOMPUTED_DIR`). In a JupyterLab workstation open the workstation URL with `/proxy/8080/`
  appended (`environment/postInstall` installs `jupyter-server-proxy`); a custom-app workstation can
  run `code/run_app.sh` directly.
- **Environment**: Python 3.11 or newer base image; `environment/postInstall` installs
  `requirements.txt`. No atlas download is needed at runtime because the CCF meshes are precomputed.
