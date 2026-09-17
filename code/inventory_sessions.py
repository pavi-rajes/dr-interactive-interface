#!/usr/bin/env python3
"""Inventory all NWB sessions in the data directory.

Reads only small metadata datasets with h5py (never units/spike_times) and writes
    <out_dir>/sessions_manifest.json   (detailed, keyed by session_id)
    <out_dir>/sessions_manifest.csv    (one flat row per session)

Usage:
    python code/inventory_sessions.py [--data-dir DIR] [--out-dir DIR]
Defaults come from code/config.py (env vars DR_DATA_DIR / DR_PRECOMPUTED_DIR).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def _scalar(ds) -> object:
    """Read a scalar HDF5 dataset and convert to a plain Python type."""
    v = ds[()]
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _str_col(grp, name) -> np.ndarray:
    return grp[name][:].astype(str)


def _py(v):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, bytes):
        return v.decode()
    return v


# ----------------------------------------------------------------------------- core
def inventory_session(path: Path) -> dict:
    sid = config.session_id_from_path(path)
    out: dict = {"session_id": sid, "file": str(path), "file_size_gb": round(path.stat().st_size / 1e9, 3)}
    with h5py.File(path, "r") as f:
        g = f["general"]
        subj = g["subject"]
        out["nwb_session_id"] = _scalar(g["session_id"]) if "session_id" in g else None
        out["identifier"] = _scalar(f["identifier"])
        out["session_start_time"] = _scalar(f["session_start_time"])
        out["session_description"] = _scalar(f["session_description"])
        out["subject"] = {k: _scalar(subj[k]) for k in ("subject_id", "genotype", "sex", "age", "strain", "species") if k in subj}

        # ---- units (metadata only)
        u = f["units"]
        n_units = int(u["id"].shape[0])
        qc = u["default_qc"][:].astype(bool)
        structure = _str_col(u, "structure")
        ccf_ap = u["ccf_ap"][:]
        structures: dict[str, dict] = {}
        for s in np.unique(structure):
            m = structure == s
            structures[str(s)] = {"n_units": int(m.sum()), "n_units_qc": int((m & qc).sum())}
        out["units"] = {
            "n_units": n_units,
            "n_units_qc": int(qc.sum()),
            "n_units_with_ccf": int(np.isfinite(ccf_ap).sum()),
            "n_structures": len(structures),
            "structures": structures,
        }

        # ---- trials
        t = f["intervals"]["trials"]
        block_index = t["block_index"][:]
        modality = _str_col(t, "rewarded_modality")
        blocks = np.unique(block_index)
        seq = [str(modality[block_index == b][0]) for b in blocks]
        out["trials"] = {
            "n_trials": int(t["id"].shape[0]),
            "n_blocks": int(len(blocks)),
            "block_modality_sequence": seq,
            "n_block_switches": int(t["is_block_switch"][:].astype(bool).sum()),
            "has_opto": bool(t["is_opto"][:].astype(bool).any()),
            "n_opto_trials": int(t["is_opto"][:].astype(bool).sum()),
            "stim_names": sorted(set(_str_col(t, "stim_name").tolist())),
            "first_stim_start_time": float(t["stim_start_time"][:].min()),
            "last_stim_start_time": float(t["stim_start_time"][:].max()),
        }

        # ---- epochs
        e = f["intervals"]["epochs"]
        names = _str_col(e, "script_name")
        starts, stops = e["start_time"][:], e["stop_time"][:]
        out["epochs"] = [
            {"script_name": str(n), "start_time": float(a), "stop_time": float(b)}
            for n, a, b in zip(names, starts, stops)
        ]
        task = [ep for ep in out["epochs"] if ep["script_name"] == config.TASK_SCRIPT_NAME]
        out["task_epoch_start"] = task[0]["start_time"] if task else None
        out["task_epoch_stop"] = task[0]["stop_time"] if task else None
        out["n_task_epochs"] = len(task)

        # ---- performance
        if "performance" in f["intervals"]:
            p = f["intervals"]["performance"]
            pm = _str_col(p, "rewarded_modality")
            out["performance"] = [
                {
                    "block_index": int(p["block_index"][i]),
                    "rewarded_modality": str(pm[i]),
                    "cross_modality_dprime": float(p["cross_modality_dprime"][i]),
                    "hit_rate": float(p["hit_rate"][i]),
                    "false_alarm_rate": float(p["false_alarm_rate"][i]),
                }
                for i in range(p["id"].shape[0])
            ]
        else:
            out["performance"] = []

        # ---- electrodes
        el = g["extracellular_ephys"]["electrodes"]
        out["electrodes"] = {
            "n_electrodes": int(el["id"].shape[0]),
            "n_probes": int(len(np.unique(_str_col(el, "group_name")))),
            "probes": sorted(set(_str_col(el, "group_name").tolist())),
        }
    return _py(out)


def flat_row(s: dict) -> dict:
    if "error" in s:
        return {"session_id": s["session_id"], "error": s["error"]}
    return {
        "session_id": s["session_id"],
        "subject_id": s["subject"].get("subject_id"),
        "date": s["session_start_time"][:10],
        "genotype": s["subject"].get("genotype"),
        "sex": s["subject"].get("sex"),
        "n_units": s["units"]["n_units"],
        "n_units_qc": s["units"]["n_units_qc"],
        "n_structures": s["units"]["n_structures"],
        "n_trials": s["trials"]["n_trials"],
        "n_blocks": s["trials"]["n_blocks"],
        "block_modality_sequence": "-".join(s["trials"]["block_modality_sequence"]),
        "has_opto": s["trials"]["has_opto"],
        "task_epoch_start": s["task_epoch_start"],
        "task_epoch_stop": s["task_epoch_stop"],
        "n_probes": s["electrodes"]["n_probes"],
        "file_size_gb": s["file_size_gb"],
        "error": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.PRECOMPUTED_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = config.list_sessions(args.data_dir)
    if not sessions:
        print(f"No .nwb files found in {args.data_dir}", file=sys.stderr)
        return 1

    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "task_script_name": config.TASK_SCRIPT_NAME,
        "sessions": {},
    }
    t0 = time.time()
    for path in sessions:
        sid = config.session_id_from_path(path)
        ts = time.time()
        try:
            manifest["sessions"][sid] = inventory_session(path)
            print(f"[ok]  {sid}  ({time.time() - ts:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            manifest["sessions"][sid] = {"session_id": sid, "file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            print(f"[ERR] {sid}: {exc}", file=sys.stderr)
            traceback.print_exc()

    json_path = args.out_dir / "sessions_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2))

    rows = [flat_row(s) for s in manifest["sessions"].values()]
    fields = list(flat_row(next(s for s in manifest["sessions"].values() if "error" not in s)).keys()) if any("error" not in s for s in manifest["sessions"].values()) else ["session_id", "error"]
    csv_path = args.out_dir / "sessions_manifest.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary table
    hdr = f"{'session_id':<22}{'subject':<9}{'units':>7}{'qc':>6}{'trials':>8}{'blocks':>7}  {'seq':<24}{'opto':>6}{'probes':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("error"):
            print(f"{r['session_id']:<22} ERROR: {r['error']}")
            continue
        print(f"{r['session_id']:<22}{r['subject_id']:<9}{r['n_units']:>7}{r['n_units_qc']:>6}{r['n_trials']:>8}{r['n_blocks']:>7}  {r['block_modality_sequence']:<24}{str(r['has_opto']):>6}{r['n_probes']:>7}")
    print(f"\nWrote {json_path} and {csv_path} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
