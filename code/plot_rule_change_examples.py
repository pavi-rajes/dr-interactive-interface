#!/usr/bin/env python3
"""Plot switch-aligned firing profiles for the 4 strongest rule-updating units."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precomputed-dir", type=Path, default=config.PRECOMPUTED_DIR)
    a = ap.parse_args()
    t = pd.read_parquet(a.precomputed_dir / "rule_change_units.parquet")
    prof = pd.read_parquet(a.precomputed_dir / "rule_change_profiles.parquet")
    both = t[(t["context"] == "both") & t["kw_is_instruction_higher"]].copy()
    both["pmin"] = both[["bl_kw_p_perm", "ev_kw_p_perm"]].min(axis=1)
    both["alr"] = np.where(both["ev_kw_is_rule_updating"] & ~both["bl_kw_is_rule_updating"], both["ev_kw_log2_ratio"].abs(), both["bl_kw_log2_ratio"].abs())
    top = both.sort_values(["pmin", "alr"], ascending=[True, False]).head(4)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), layout="tight", width_ratios=(2.2, 1), sharey="row")
    for i, (_, r) in enumerate(top.iterrows()):
        win = "evoked" if (r["ev_kw_is_rule_updating"] and not r["bl_kw_is_rule_updating"]) else "baseline"
        w = "ev" if win == "evoked" else "bl"
        for ax, align in zip(axes[i], ("block_start", "block_end")):
            for ctx, col in (("aud", "#d95f02"), ("vis", "#1b6fb3")):
                p = prof[(prof["session_id"] == r["session_id"]) & (prof["unit_index"] == r["unit_index"]) & (prof["context"] == ctx)
                         & (prof["align"] == align) & (prof["window"] == win)].sort_values("rel_trial")
                ax.plot(p["rel_trial"], p["mean_rate"], marker="o", ms=3, color=col, label=f"{ctx} blocks (n={int(p['n_blocks'].max())})")
            if align == "block_start":
                ax.axvspan(-0.5, 4.5, color="gray", alpha=0.18, label="instruction")
                ax.axvspan(4.5, 14.5, color="#c0392b", alpha=0.12, label="early")
                ax.set_xlabel("trial in block (from block start)")
                ax.set_title(f"{r['session_id']} unit {int(r['unit_index'])} {r['structure']} | {win} | KW H={r[w + '_kw_h']:.1f} "
                             f"p_perm={r[w + '_kw_p_perm']:.2g} q={r[w + '_kw_q_bh']:.2g} | log2(instr/rest)={r[w + '_kw_log2_ratio']:+.2f}", fontsize=8)
                ax.set_ylabel(f"{win} rate (Hz)")
            else:
                ax.axvspan(-10.5, -0.5, color="#555555", alpha=0.15, label="late")
                ax.set_xlabel("trial in block (from block end)")
            if i == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Rule-updating units (KW instruction > rest): mean rate per within-block position, averaged over blocks of each context", fontsize=10)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
