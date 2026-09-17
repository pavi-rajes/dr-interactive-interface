#!/usr/bin/env python3
"""Plot per-trial baseline rates for the 4 most robust context-shift units (Results figure)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from analyze_context_baseline_shift import load_session, per_trial_baseline_rates, rate_matrix, select_trials  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precomputed-dir", type=Path, default=config.PRECOMPUTED_DIR)
    a = ap.parse_args()
    P = json.loads((a.precomputed_dir / "context_baseline_shifts_summary.json").read_text())["params"]
    t = pd.read_parquet(a.precomputed_dir / "context_baseline_shifts.parquet")
    top = t[t["is_robust"]].assign(abs_lr=lambda d: d["log2_ratio"].abs()).sort_values(["p_perm", "abs_lr"], ascending=[True, False]).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        units, trials, aligned = load_session(a.precomputed_dir, r["session_id"])
        sel = select_trials(trials, P["exclude_first_n"])
        R = rate_matrix(per_trial_baseline_rates(aligned, sel, units, tuple(P["baseline_window"])), units, sel)
        row = int(np.where(units["unit_index"].to_numpy() == r["unit_index"])[0][0])
        x = sel["trial_index"].to_numpy(); y = R[row]
        aud = (sel["rewarded_modality"] == "aud").to_numpy()
        ax.scatter(x[aud], y[aud], s=8, color="tab:orange", label="aud block")
        ax.scatter(x[~aud], y[~aud], s=8, color="tab:blue", label="vis block")
        for b, g in sel.groupby("block_index"):
            ax.axvline(g["trial_index"].min(), color="gray", lw=0.6, ls="--")
            ax.hlines(y[(sel["block_index"] == b).to_numpy()].mean(), g["trial_index"].min(), g["trial_index"].max(), color="k", lw=1.5)
        ax.set_title(f"{r['session_id']} unit {int(r['unit_index'])} {r['structure']} | {r['shift_direction']} "
                     f"log2={r['log2_ratio']:.2f} p_perm={r['p_perm']:.2g} q={r['q_bh']:.2g} p_mwu={r['p_mwu']:.1g}", fontsize=8)
        ax.set_xlabel("trial index"); ax.set_ylabel("baseline rate (Hz)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Context-dependent baseline shifts: per-trial pre-stimulus rate, block means in black", fontsize=10)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
