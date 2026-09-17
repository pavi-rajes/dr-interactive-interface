#!/usr/bin/env python3
"""Validate precomputed aligned spikes for one session against the raw NWB file.

Usage: python code/validate_aligned_spikes.py [--session SID] [--n-trials 20] [--seed 0]
Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="759434_2025-02-04")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.PRECOMPUTED_DIR)
    a = ap.parse_args()

    sdir = a.out_dir / a.session
    manifest = json.loads((a.out_dir / "precompute_manifest.json").read_text())
    pre, post = manifest["params"]["window"]
    units = pd.read_parquet(sdir / "units.parquet")
    trials = pd.read_parquet(sdir / "trials.parquet")
    aligned = pd.read_parquet(sdir / "aligned_spikes.parquet")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    rng = np.random.default_rng(a.seed)
    with h5py.File(a.data_dir / f"{a.session}.nwb", "r") as f:
        st = f["units"]["spike_times"]
        idx = f["units"]["spike_times_index"][:].astype(np.int64)
        starts = np.concatenate([[0], idx[:-1]])
        stim = f["intervals"]["trials"]["stim_start_time"][:]
        n_units_nwb = f["units"]["id"].shape[0]
        qc_nwb = int(f["units"]["default_qc"][:].astype(bool).sum())
        n_trials_nwb = f["intervals"]["trials"]["id"].shape[0]
        fr_nwb = f["units"]["firing_rate"][:]

        # (a) brute-force recount for 3 units x N random trials
        picks = [units["unit_index"].iloc[0], units["unit_index"].iloc[len(units) // 2], units["unit_index"].iloc[-1]]
        tsel = rng.choice(len(stim), size=min(a.n_trials, len(stim)), replace=False)
        all_ok, mism = True, []
        for ui in picks:
            spk = st[starts[ui]:idx[ui]]
            sub = aligned[aligned["unit_index"] == ui]
            for ti in tsel:
                s0 = stim[ti]
                brute = spk[(spk >= s0 + pre) & (spk < s0 + post)] - s0
                got = np.sort(sub.loc[sub["trial_index"] == ti, "t_rel"].to_numpy(dtype=np.float64))
                if len(brute) != len(got) or not np.allclose(np.sort(brute), got, atol=1e-4):
                    all_ok = False
                    mism.append((int(ui), int(ti), len(brute), len(got)))
        check("(a) brute-force recount matches for 3 units x %d trials" % len(tsel), all_ok, f"units {list(map(int, picks))}" + (f" mismatches {mism[:5]}" if mism else ""))

    # (b) window bounds (t_rel is float32; allow equality at POST due to rounding)
    tmin, tmax = float(aligned["t_rel"].min()), float(aligned["t_rel"].max())
    check("(b) t_rel within window", tmin >= pre and tmax <= post, f"min {tmin:.4f} max {tmax:.4f} window [{pre}, {post}]")

    # (c) table sizes
    check("(c1) trials rows == NWB trials", len(trials) == n_trials_nwb, f"{len(trials)} vs {n_trials_nwb}")
    check("(c2) units rows == NWB QC-pass units", len(units) == qc_nwb, f"{len(units)} vs {qc_nwb} (of {n_units_nwb})")
    check("(c3) no NaN unit_id", units["unit_id"].notna().all() and (units["unit_id"] != "").all())
    check("(c4) n_spikes_aligned matches aligned table", np.array_equal(
        units.set_index("unit_index")["n_spikes_aligned"].sort_index().to_numpy(),
        aligned.groupby("unit_index").size().reindex(units["unit_index"].sort_values(), fill_value=0).to_numpy()))

    # (d) pre-stimulus rate sanity for one unit
    ui = picks[1]
    sub = aligned[(aligned["unit_index"] == ui) & (aligned["t_rel"] >= -1.5) & (aligned["t_rel"] < -0.06)]
    rate = len(sub) / (len(trials) * (1.5 - 0.06))
    fr = float(fr_nwb[ui]); tfr = float(units.loc[units["unit_index"] == ui, "task_firing_rate"].iloc[0])
    ratio = rate / tfr if tfr > 0 else np.inf
    check("(d) pre-stim rate plausible vs task rate", 0.2 <= ratio <= 5.0,
          f"unit {int(ui)}: pre-stim {rate:.2f} Hz, task epoch {tfr:.2f} Hz, NWB session firing_rate {fr:.2f} Hz")

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed for {a.session}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
