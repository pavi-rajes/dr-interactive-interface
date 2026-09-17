#!/usr/bin/env python3
"""Precompute stimulus-aligned spike tables for the Shiny viewer.

For every session in DATA_DIR and every unit passing default_qc (or all units with
--all-units), extract spike times relative to each trial's stim_start_time within
[PRE, POST] seconds, and write per session into <out-dir>/<session_id>/:

    units.parquet           one row per unit (metadata, QC, CCF coords, task rates)
    trials.parquet          one row per trial (block/context labels, outcome flags,
                            event times relative to stim onset)
    aligned_spikes.parquet  columns unit_index (int32), trial_index (int16),
                            t_rel (float32, seconds relative to stim onset)

plus cross-session <out-dir>/units_all.parquet and <out-dir>/precompute_manifest.json.

Usage:
    python code/precompute_aligned_spikes.py [--data-dir D] [--out-dir D]
        [--sessions SID ...] [--window PRE POST] [--all-units] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SCRIPT_VERSION = "1.0"

TRIAL_BOOL_COLS = [
    "is_block_switch", "is_vis_stim", "is_aud_stim", "is_target", "is_nontarget",
    "is_go", "is_nogo", "is_catch", "is_hit", "is_miss", "is_false_alarm",
    "is_correct_reject", "is_response", "is_rewarded", "is_opto", "is_repeat",
    "is_instruction", "is_correct",
]
# (nwb column, output column) for event times converted to stim-relative seconds
TRIAL_REL_COLS = [
    ("quiescent_start_time", "quiescent_start_rel"),
    ("quiescent_stop_time", "quiescent_stop_rel"),
    ("stim_stop_time", "stim_stop_rel"),
    ("response_window_start_time", "response_window_start_rel"),
    ("response_window_stop_time", "response_window_stop_rel"),
    ("response_time", "response_time_rel"),
    ("reward_time", "reward_time_rel"),
]
UNIT_COLS = [
    "unit_id", "structure", "location", "ccf_ap", "ccf_dv", "ccf_ml", "default_qc",
    "firing_rate", "presence_ratio", "isi_violations_ratio", "amplitude_cutoff", "snr",
    "half_width", "peak_to_valley", "num_spikes", "electrode_group_name", "peak_channel",
    "activity_drift", "is_not_drift",
]


def _col(grp, name) -> np.ndarray:
    v = grp[name][:]
    if v.dtype.kind in ("O", "S"):
        return v.astype(str)
    return v


def load_trials(f: h5py.File) -> pd.DataFrame:
    t = f["intervals"]["trials"]
    n = t["id"].shape[0]
    stim_start = t["stim_start_time"][:].astype(float)
    df = pd.DataFrame({"trial_index": np.arange(n, dtype=np.int16)})
    nwb_ti = t["trial_index"][:]
    if not np.array_equal(nwb_ti, np.arange(n)):
        df["trial_index_nwb"] = nwb_ti
    df["block_index"] = t["block_index"][:].astype(np.int16)
    df["trial_index_in_block"] = t["trial_index_in_block"][:].astype(np.int16)
    df["rewarded_modality"] = _col(t, "rewarded_modality")
    df["stim_name"] = _col(t, "stim_name")
    for c in TRIAL_BOOL_COLS:
        df[c] = t[c][:].astype(bool)
    df["stim_start_time"] = stim_start
    for src, dst in TRIAL_REL_COLS:
        df[dst] = (t[src][:].astype(float) - stim_start).astype(np.float32)
    # switch direction per block: "<prev>_to_<cur>", NaN for first block
    first_mod = df.groupby("block_index")["rewarded_modality"].first()
    direction = {}
    prev = None
    for b in sorted(first_mod.index):
        direction[b] = (f"{prev}_to_{first_mod[b]}" if prev is not None else np.nan)
        prev = first_mod[b]
    df["switch_direction"] = df["block_index"].map(direction)
    return df


def load_units(f: h5py.File, session_id: str, qc_only: bool) -> pd.DataFrame:
    u = f["units"]
    n = u["id"].shape[0]
    df = pd.DataFrame({"session_id": session_id, "unit_index": np.arange(n, dtype=np.int32)})
    for c in UNIT_COLS:
        df[c] = _col(u, c)
    df["default_qc"] = df["default_qc"].astype(bool)
    if qc_only:
        df = df[df["default_qc"]].reset_index(drop=True)
    return df


def task_epoch(f: h5py.File) -> tuple[float, float]:
    e = f["intervals"]["epochs"]
    names = _col(e, "script_name")
    idx = np.where(names == config.TASK_SCRIPT_NAME)[0]
    if len(idx) == 0:
        raise RuntimeError(f"no epoch with script_name == {config.TASK_SCRIPT_NAME}")
    i = idx[0]
    return float(e["start_time"][i]), float(e["stop_time"][i])


def align_spikes(f: h5py.File, units_df: pd.DataFrame, trials_df: pd.DataFrame,
                 window: tuple[float, float], epoch: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (aligned DataFrame, units_df with n_spikes_task/task_firing_rate/n_spikes_aligned)."""
    pre, post = window
    st = f["units"]["spike_times"]
    idx = f["units"]["spike_times_index"][:].astype(np.int64)
    starts = np.concatenate([[0], idx[:-1]])
    stim = trials_df["stim_start_time"].to_numpy(dtype=float)
    lo_edges, hi_edges = stim + pre, stim + post
    trial_idx = trials_df["trial_index"].to_numpy(dtype=np.int16)
    ep0, ep1 = epoch

    u_parts, t_parts, r_parts = [], [], []
    n_task = np.zeros(len(units_df), dtype=np.int64)
    n_aligned = np.zeros(len(units_df), dtype=np.int64)
    for k, ui in enumerate(units_df["unit_index"].to_numpy()):
        spk = st[starts[ui]:idx[ui]]
        n_task[k] = np.searchsorted(spk, ep1) - np.searchsorted(spk, ep0)
        lo = np.searchsorted(spk, lo_edges)
        hi = np.searchsorted(spk, hi_edges)
        counts = hi - lo
        total = int(counts.sum())
        n_aligned[k] = total
        if total == 0:
            continue
        # gather indices for all trials at once
        offsets = np.repeat(lo - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        gather = np.arange(total) + offsets
        rel = spk[gather] - np.repeat(stim, counts)
        u_parts.append(np.full(total, ui, dtype=np.int32))
        t_parts.append(np.repeat(trial_idx, counts))
        r_parts.append(rel.astype(np.float32))

    aligned = pd.DataFrame({
        "unit_index": np.concatenate(u_parts) if u_parts else np.array([], dtype=np.int32),
        "trial_index": np.concatenate(t_parts) if t_parts else np.array([], dtype=np.int16),
        "t_rel": np.concatenate(r_parts) if r_parts else np.array([], dtype=np.float32),
    })
    units_df = units_df.copy()
    units_df["n_spikes_task"] = n_task
    units_df["task_firing_rate"] = n_task / (ep1 - ep0)
    units_df["n_spikes_aligned"] = n_aligned
    return aligned, units_df


def process_session(path: Path, out_dir: Path, window: tuple[float, float], qc_only: bool) -> dict:
    sid = config.session_id_from_path(path)
    sdir = out_dir / sid
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with h5py.File(path, "r") as f:
        trials = load_trials(f)
        units = load_units(f, sid, qc_only)
        epoch = task_epoch(f)
        aligned, units = align_spikes(f, units, trials, window, epoch)
    aligned = aligned.sort_values(["unit_index", "trial_index"], kind="stable").reset_index(drop=True)
    trials.to_parquet(sdir / "trials.parquet", index=False)
    units.to_parquet(sdir / "units.parquet", index=False)
    aligned.to_parquet(sdir / "aligned_spikes.parquet", index=False, compression="zstd")
    dt = time.time() - t0
    sizes = {p.name: p.stat().st_size for p in sdir.glob("*.parquet")}
    info = {
        "session_id": sid, "n_units": int(len(units)), "n_trials": int(len(trials)),
        "n_aligned_rows": int(len(aligned)), "task_epoch": list(epoch),
        "runtime_s": round(dt, 1), "file_sizes_bytes": sizes,
    }
    print(f"[ok] {sid}: {info['n_units']} units, {info['n_trials']} trials, "
          f"{info['n_aligned_rows']:,} aligned spikes, {dt:.1f}s, "
          f"{sizes.get('aligned_spikes.parquet', 0) / 1e6:.1f} MB")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.PRECOMPUTED_DIR)
    ap.add_argument("--sessions", nargs="+", default=None, help="subset of session ids")
    ap.add_argument("--window", nargs=2, type=float, default=(-2.0, 3.0), metavar=("PRE", "POST"))
    ap.add_argument("--all-units", action="store_true", help="include units failing default_qc")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    window = (float(args.window[0]), float(args.window[1]))
    qc_only = not args.all_units

    paths = config.list_sessions(args.data_dir)
    if args.sessions:
        paths = [p for p in paths if config.session_id_from_path(p) in set(args.sessions)]
    if not paths:
        print("no sessions selected", file=sys.stderr)
        return 1

    t0 = time.time()
    infos = []
    for p in paths:
        sid = config.session_id_from_path(p)
        sdir = args.out_dir / sid
        if not args.overwrite and all((sdir / n).exists() for n in ("units.parquet", "trials.parquet", "aligned_spikes.parquet")):
            print(f"[skip] {sid}: outputs exist (use --overwrite)")
            continue
        try:
            infos.append(process_session(p, args.out_dir, window, qc_only))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {sid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            infos.append({"session_id": sid, "error": f"{type(exc).__name__}: {exc}"})

    # cross-session outputs (over everything present on disk, not just this run)
    unit_files = sorted(args.out_dir.glob("*/units.parquet"))
    if unit_files:
        all_units = pd.concat([pd.read_parquet(p) for p in unit_files], ignore_index=True)
        all_units.to_parquet(args.out_dir / "units_all.parquet", index=False)
        print(f"units_all.parquet: {len(all_units)} units from {len(unit_files)} sessions")

    manifest_path = args.out_dir / "precompute_manifest.json"
    prev = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sessions = prev.get("sessions", {}) if prev.get("params") == {"window": list(window), "qc_only": qc_only} else {}
    for i in infos:
        sessions[i["session_id"]] = i
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "data_dir": str(args.data_dir),
        "params": {"window": list(window), "qc_only": qc_only},
        "total_runtime_s_this_run": round(time.time() - t0, 1),
        "sessions": sessions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}; total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
