---
slug: project-scaffold-and-inventory
status: done
plan_review_attempts: 1
---

# Goal
Establish the reproducible project layout and a data inventory. Create `code/` and `data/precomputed/` directories, and a small config module (e.g. `code/config.py`) that exposes `DATA_DIR` (defaulting to `nwb_sessions/` at the repo root, overridable via an environment variable) and `PRECOMPUTED_DIR` (`data/precomputed/`). Add a dependency manifest (`requirements.txt` or `pyproject.toml`) listing pynwb/h5py, numpy, pandas, pyarrow, scipy, plotly, and shiny.

Write a script `code/inventory_sessions.py` that iterates over every `.nwb` file in `DATA_DIR`, reads with h5py (to avoid loading spike data), and writes `data/precomputed/sessions_manifest.json` (or parquet) with, per session: subject id, session date, number of units, number passing `default_qc`, unique structures with unit counts, number of trials, number of blocks, rewarded-modality sequence, presence of opto trials, and the DynamicRouting task epoch start/stop. The script must be idempotent and runnable as `python code/inventory_sessions.py`.

# Instructions

All paths are relative to the repo root `/Users/pavir/Code/dr-interactive-interface`. Use the existing interpreter `/opt/miniconda3/bin/python3` (Python 3.13; pynwb 3.1.3, h5py 3.15.1, shiny 1.7.0 already installed). Use the Bash tool for file creation and running scripts.

1. **Create directory layout.** `mkdir -p code data/precomputed work/project-scaffold-and-inventory/data`. Add `data/precomputed/.gitkeep`.

2. **Write `code/config.py`.** Contents:
   - `REPO_ROOT = Path(__file__).resolve().parents[1]`
   - `DATA_DIR = Path(os.environ.get("DR_DATA_DIR", REPO_ROOT / "nwb_sessions"))`
   - `PRECOMPUTED_DIR = Path(os.environ.get("DR_PRECOMPUTED_DIR", REPO_ROOT / "data" / "precomputed"))`
   - `TASK_SCRIPT_NAME = "DynamicRouting1"` (the epochs `script_name` value marking the task epoch)
   - helper `list_sessions() -> list[Path]` returning sorted `DATA_DIR.glob("*.nwb")`
   - helper `session_id_from_path(p) -> str` returning the file stem (e.g. `759434_2025-02-04`).
   Module must import cleanly with `python3 -c "import sys; sys.path.insert(0,'code'); import config; print(config.DATA_DIR)"`.

3. **Write `requirements.txt`** at repo root listing: `pynwb>=3.1`, `h5py>=3.10`, `numpy`, `pandas`, `pyarrow`, `scipy`, `plotly`, `shiny>=1.7`. Then run `/opt/miniconda3/bin/python3 -m pip install -r requirements.txt` and confirm all imports succeed with a one-line `python3 -c "import pynwb,h5py,numpy,pandas,pyarrow,scipy,plotly,shiny"`.

4. **Write `code/inventory_sessions.py`.** Uses h5py only (never load `units/spike_times`). For each file from `config.list_sessions()` read:
   - `general/session_id`, `general/subject/subject_id`, `general/subject/genotype`, `general/subject/sex`, `general/subject/age`, `session_start_time`, `identifier`
   - `units`: `n_units = len(units/id)`, `n_units_qc = default_qc.sum()`, `structures`: dict of structure -> {n_units, n_units_qc} from `units/structure` and `units/default_qc`; count of units with non-NaN `ccf_ap`
   - `intervals/trials`: `n_trials`, `n_blocks = len(unique(block_index))`, `block_modality_sequence` = rewarded_modality of first trial in each block (list of str), `n_block_switches = is_block_switch.sum()`, `has_opto = is_opto.any()`, `stim_names` = sorted unique `stim_name`
   - `intervals/epochs`: rows where `script_name == config.TASK_SCRIPT_NAME` -> `task_epoch_start`, `task_epoch_stop` (floats); also list all epoch script names in order
   - `intervals/performance`: per block `rewarded_modality`, `cross_modality_dprime`, `hit_rate`, `false_alarm_rate` (as list of dicts)
   - `general/extracellular_ephys/electrodes`: `n_electrodes`, `n_probes = len(unique(group_name))`.
   Decode all byte strings to str; convert numpy scalars to Python types. Write `data/precomputed/sessions_manifest.json` (pretty-printed, keyed by session_id, with a top-level `generated_at` ISO timestamp and `data_dir`), and a flat `data/precomputed/sessions_manifest.csv` with one row per session (session_id, subject_id, date, genotype, n_units, n_units_qc, n_trials, n_blocks, has_opto, task_epoch_start, task_epoch_stop, n_probes). Wrap per-session processing in try/except that logs the error and continues, recording `error` in that session's entry. Print a summary table to stdout. Support `--data-dir` and `--out-dir` CLI overrides via argparse (defaults from config).

5. **Run it.** `cd` to repo root and run `/opt/miniconda3/bin/python3 code/inventory_sessions.py`. Expected: 5 sessions processed with no errors; the session `759434_2025-02-04` should show 2285 units, 744 QC-pass, 545 trials, 6 blocks, task epoch ~2155.7 to ~5796.8 s. Record wall time (expect under a few minutes since spike data is not read). Copy the printed summary into the Results section.

6. **Verify reproducibility.** Run the script a second time and confirm the JSON is identical apart from `generated_at` (e.g. `diff <(jq 'del(.generated_at)' ...)` or a Python comparison). Run `python3 code/inventory_sessions.py --data-dir nwb_sessions --out-dir work/project-scaffold-and-inventory/data` to confirm the overrides work, and keep that copy as the work artifact.

7. **Write `code/README.md`** briefly describing config variables (`DR_DATA_DIR`, `DR_PRECOMPUTED_DIR`), the script and how to run it, and the output files.

Expected artifacts: `code/config.py`, `code/inventory_sessions.py`, `code/README.md`, `requirements.txt`, `data/precomputed/sessions_manifest.json`, `data/precomputed/sessions_manifest.csv`, and a copy of the manifest under `work/project-scaffold-and-inventory/data/`.

# Results

## Summary
The project scaffold is in place: `code/`, `data/precomputed/`, a `config.py` exposing `DATA_DIR`/`PRECOMPUTED_DIR` (env-overridable via `DR_DATA_DIR`/`DR_PRECOMPUTED_DIR`) and `TASK_SCRIPT_NAME`, a `requirements.txt` whose packages all import, and `code/inventory_sessions.py`, which inventoried all 5 NWB sessions with no errors in about 0.1 s total (metadata only via h5py). The manifest was regenerated twice and is byte-identical apart from `generated_at`; the `--data-dir`/`--out-dir` overrides work and were used to produce the copy under `work/project-scaffold-and-inventory/data/`. Expected values for session 759434_2025-02-04 (2285 units, 744 QC-pass, 545 trials, 6 blocks, task epoch 2155.7-5796.8 s) matched exactly.

## Artifacts
- `code/config.py` — data-location config module and session helpers.
- `code/inventory_sessions.py` — reproducible session inventory script (argparse; h5py only).
- `code/README.md` — documentation of config variables, scripts, and outputs.
- `requirements.txt` — dependency manifest (pynwb, h5py, numpy, pandas, pyarrow, scipy, plotly, shiny).
- `data/precomputed/sessions_manifest.json` — detailed manifest keyed by session id (subject, units/QC/structures with counts, trials/blocks/modality sequence/opto, epochs, task epoch bounds, per-block performance, electrodes/probes).
- `data/precomputed/sessions_manifest.csv` — flat one-row-per-session summary.
- `data/sessions_manifest.json`, `data/sessions_manifest.csv` (under this work folder) — copy produced via CLI overrides.

## Session summary (from the script's stdout)

| session_id | subject | genotype | units | qc | trials | blocks | modality sequence | opto | probes | structures |
|---|---|---|---|---|---|---|---|---|---|---|
| 664851_2023-11-15 | 664851 | Pvalb-IRES-Cre;Ai32 | 3062 | 1118 | 534 | 6 | aud-vis-aud-vis-aud-vis | False | 5 | 24 |
| 668755_2023-08-31 | 668755 | wt/wt | 2878 | 1184 | 524 | 6 | vis-aud-vis-aud-vis-aud | False | 6 | 34 |
| 713655_2024-08-09 | 713655 | Sst-IRES-Cre;Ai32 | 3577 | 1666 | 515 | 6 | aud-vis-aud-vis-aud-vis | False | 5 | 21 |
| 742903_2024-10-22 | 742903 | Vip-IRES-Cre;Ai32 | 4446 | 1973 | 538 | 6 | aud-vis-aud-vis-aud-vis | False | 6 | 23 |
| 759434_2025-02-04 | 759434 | VGAT-ChR2-YFP | 2285 | 744 | 545 | 6 | vis-aud-vis-aud-vis-aud | False | 5 | 19 |

Totals: 16,248 units, 6,685 QC-pass. No session has opto trials. Every session has exactly one `DynamicRouting1` epoch (~1 h) and 6 blocks with alternating rewarded modality. Cross-modality d' per block ranges roughly 0.8 to 3.9 (all stored in the JSON `performance` field).

## Step-by-step
1. Created `code/`, `data/precomputed/` (with `.gitkeep`), and `work/project-scaffold-and-inventory/data/`.
2. Wrote `code/config.py` as specified; verified it imports and lists 5 sessions.
3. Wrote `requirements.txt`; `pip install -r requirements.txt` succeeded (pandas 2.3.3, pyarrow 23.0.1, plotly 6.7.0 present); the one-line import check passed.
4. Wrote `code/inventory_sessions.py` reading exactly the planned HDF5 paths, with try/except per session and `--data-dir`/`--out-dir` overrides. Small addition beyond the plan: also records file size, strain/species, n_opto_trials, first/last stim time, and probe names, since they were free to collect.
5. Ran the script: 5/5 sessions ok, 0.1 s wall time; expected values for 759434_2025-02-04 matched. Minor cosmetic fix to the stdout table column widths after the first run.
6. Reproducibility: second run identical apart from `generated_at` (checked with a Python dict comparison). Override run wrote the copy into `work/project-scaffold-and-inventory/data/`.
7. Wrote `code/README.md`.

Deviations: none material. The reviewer's note about using the full interpreter path was followed throughout.

# Assessment

## Verdict
accomplished

## Reasoning
Every element of the Goal is present and verified. `code/config.py` imports cleanly, exposes `DATA_DIR`, `PRECOMPUTED_DIR`, and `TASK_SCRIPT_NAME`, honours the `DR_DATA_DIR` env override (tested), and provides `list_sessions()` (returns 5) and `session_id_from_path()`. `requirements.txt` lists all eight required packages and the one-line import check succeeds. `code/inventory_sessions.py` reads only metadata via h5py, supports `--data-dir`/`--out-dir`, and wraps per-session processing in try/except. I re-ran it into a scratch directory: 5/5 sessions ok, 0.16 s wall time, and the resulting JSON is identical to `data/precomputed/sessions_manifest.json` apart from `generated_at`, confirming the idempotency claim. The manifest carries every field the Goal asks for (subject id, date, n_units, n_units_qc, structures with counts, n_trials, n_blocks, modality sequence, opto flag, task epoch start/stop) plus per-block performance and probe counts.

Spot-checks against the artifacts and directly against the NWB file agree with the Results: 759434_2025-02-04 has 2285 units / 744 QC / 545 trials / 6 blocks / task epoch 2155.695-5796.812 s / 5 probes / no opto; per-structure counts sum to session totals in all sessions; totals are 16,248 units and 6,685 QC-pass; cross-modality d' spans 0.78-3.92 over 30 block rows; every session has exactly one ~61 min DynamicRouting1 epoch. All seven expected artifacts exist, including the work-folder copy produced via CLI overrides (differs from the precomputed copy only in the relative `file` path, as expected from `--data-dir nwb_sessions`). The only deviation from the plan is the documented, additive set of extra fields (file size, strain/species, n_opto_trials, first/last stim time, probe names), which does not affect any required output.

## Recommended next status
done

# Review Notes
Approved: every step names a concrete file, command, or HDF5 path; all 25 referenced HDF5 paths (general/subject/*, units/{id,default_qc,structure,ccf_ap}, intervals/{trials,epochs,performance}/*, general/extracellular_ephys/electrodes/group_name) were verified read-only against 759434_2025-02-04.nwb, and the expected values in step 5 (2285 units, 744 QC-pass, 545 trials, 6 blocks, task epoch 2155.7-5796.8 s, 5 probes) match the file. Outputs are path-referenced between steps, artifact locations are explicit, the plan covers the Goal without overreach, and the h5py-only metadata approach makes the sub-minute-per-file runtime realistic.
