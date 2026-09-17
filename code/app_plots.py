"""Matplotlib plot helpers for the Shiny app (rasters, PSTHs, per-trial rates, switch profiles)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

AUD, VIS = "#d95f02", "#1b6fb3"
EARLY, LATE = "#c0392b", "#555555"


def _psth(t_rel: np.ndarray, n_trials: int, edges: np.ndarray, smooth: int = 3) -> np.ndarray:
    if n_trials == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(t_rel, bins=edges)
    rate = counts / (n_trials * (edges[1] - edges[0]))
    if smooth > 1:
        k = np.ones(smooth) / smooth
        rate = np.convolve(rate, k, mode="same")
    return rate


def raster_psth(spikes: pd.DataFrame, trials: pd.DataFrame, groups: list[tuple[str, np.ndarray, str]], window: tuple[float, float],
                bin_size: float, shade: tuple[float, float] | None = None, sort_by: str = "trial_index",
                highlight: np.ndarray | None = None, stim_dur: float = 0.5, title: str | None = None) -> plt.Figure:
    """groups: list of (label, trial_index array, colour). One raster per group (top), one PSTH panel (bottom)."""
    n = max(1, len(groups))
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3.6 * ncols + 0.4, 2.6 * nrows + 2.4), layout="tight")
    gs = GridSpec(nrows + 1, ncols, figure=fig, height_ratios=[2.0] * nrows + [1.2])
    ax_p = fig.add_subplot(gs[nrows, :])
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    tr = trials.set_index("trial_index")
    spk_by_trial = spikes.groupby("trial_index")["t_rel"]
    spk_groups = {k: v.to_numpy() for k, v in spk_by_trial}
    for gi, (label, tidx, color) in enumerate(groups):
        ax = fig.add_subplot(gs[gi // ncols, gi % ncols])
        tidx = np.asarray(tidx)
        if len(tidx):
            order = tr.loc[tidx].sort_values(sort_by).index.to_numpy()
        else:
            order = tidx
        blocks = tr.loc[order, "block_index"].to_numpy() if len(order) else np.array([])
        xs, ys, hx, hy = [], [], [], []
        for row, t in enumerate(order):
            s = spk_groups.get(t)
            if s is None:
                continue
            s = s[(s >= window[0]) & (s < window[1])]
            if highlight is not None and t in highlight:
                hx.append(s); hy.append(np.full(len(s), row))
            else:
                xs.append(s); ys.append(np.full(len(s), row))
        if xs:
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=1.2, color=color, marker="|", linewidths=0.6)
        if hx:
            ax.scatter(np.concatenate(hx), np.concatenate(hy), s=1.2, color=EARLY, marker="|", linewidths=0.6)
        if sort_by == "trial_index":  # block boundaries only make sense in chronological order
            for b in np.where(np.diff(blocks) != 0)[0]:
                ax.axhline(b + 0.5, color="k", lw=0.5, alpha=0.5)
        if shade:
            ax.axvspan(shade[0], shade[1], color="gold", alpha=0.18, lw=0)
        ax.axvspan(0, stim_dur, color="gray", alpha=0.12, lw=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(window); ax.set_ylim(-0.5, max(len(order), 1) - 0.5)
        ax.set_title(f"{label} (n={len(order)})", fontsize=8.5, color=color)
        ax.set_ylabel("trial" if gi % ncols == 0 else ""); ax.tick_params(labelsize=7.5)
        all_t = np.concatenate([spk_groups[t] for t in order if t in spk_groups]) if len(order) else np.array([])
        ax_p.plot(edges[:-1] + bin_size / 2, _psth(all_t, len(order), edges), color=color, lw=1.4, label=label)
    if shade:
        ax_p.axvspan(shade[0], shade[1], color="gold", alpha=0.18, lw=0, label="analysis window")
    ax_p.axvspan(0, stim_dur, color="gray", alpha=0.12, lw=0)
    ax_p.axvline(0, color="k", lw=0.6)
    ax_p.set_xlim(window); ax_p.set_xlabel("time from stimulus onset (s)"); ax_p.set_ylabel("rate (Hz)")
    ax_p.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False); ax_p.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def per_trial_rate_plot(spikes: pd.DataFrame, trials: pd.DataFrame, window: tuple[float, float], title: str | None = None) -> plt.Figure:
    s = spikes[(spikes["t_rel"] >= window[0]) & (spikes["t_rel"] < window[1])]
    counts = s.groupby("trial_index").size().reindex(trials["trial_index"], fill_value=0)
    rate = counts.to_numpy() / (window[1] - window[0])
    fig, ax = plt.subplots(figsize=(7, 2.6), layout="tight")
    aud = (trials["rewarded_modality"] == "aud").to_numpy()
    x = trials["trial_index"].to_numpy()
    ax.scatter(x[aud], rate[aud], s=7, color=AUD, label="aud block")
    ax.scatter(x[~aud], rate[~aud], s=7, color=VIS, label="vis block")
    for b, g in trials.groupby("block_index"):
        m = (trials["block_index"] == b).to_numpy()
        ax.axvline(g["trial_index"].min(), color="gray", lw=0.6, ls="--")
        ax.hlines(rate[m].mean(), g["trial_index"].min(), g["trial_index"].max(), color="k", lw=1.5)
    ax.set_xlabel("trial index"); ax.set_ylabel(f"rate in [{window[0]:.2f}, {window[1]:.2f}] s (Hz)")
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, frameon=False); ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def switch_profile_plot(prof_unit: pd.DataFrame, prof_region: pd.DataFrame | None = None, window: str = "baseline",
                        title: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="tight")
    for d, col in (("aud_to_vis", VIS), ("vis_to_aud", AUD)):
        p = prof_unit[prof_unit["direction"] == d].sort_values("rel_trial")
        if len(p):
            ax.plot(p["rel_trial"], p["mean_rate"], marker="o", ms=3, lw=1.2, color=col,
                    label=f"{d.replace('_', ' ')} (n={int(p['n_switches'].max())} switches)")
    if prof_region is not None and len(prof_region):
        pr = prof_region[prof_region["direction"] == "both"].sort_values("rel_trial")
        if len(pr):
            ax.plot(pr["rel_trial"], pr["mean_rate"], color="k", lw=1, alpha=0.5, label=f"{pr['structure'].iloc[0]} rule-updating units (both)")
            ax.fill_between(pr["rel_trial"], pr["mean_rate"] - pr["sem_rate"], pr["mean_rate"] + pr["sem_rate"], color="k", alpha=0.08, lw=0)
    ax.axvspan(-0.5, 4.5, color="gray", alpha=0.15, label="instruction trials")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("trials relative to block switch"); ax.set_ylabel(f"{window} rate (Hz)")
    ax.legend(fontsize=7, loc="upper right"); ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return fig


# ----------------------------------------------------------------------------- tab 2 (rule change) plots
CTX_COLOR = {"aud": AUD, "vis": VIS}
INSTR, EARLYC, LATEC = "#8c8c8c", "#c0392b", "#1c1c1c"


def _trial_classes(trials: pd.DataFrame, instruction_n: int, early_window: tuple[int, int], late_n: int) -> np.ndarray:
    """'instruction' | 'early' | 'late' | 'other' per trial (all blocks)."""
    tib = trials["trial_index_in_block"].to_numpy()
    nb = trials.groupby("block_index")["trial_index"].transform("size").to_numpy()
    cls = np.full(len(trials), "other", dtype=object)
    cls[tib < instruction_n] = "instruction"
    cls[(tib >= early_window[0]) & (tib < early_window[1])] = "early"
    cls[tib >= nb - late_n] = "late"
    return cls


def block_timeline_plot(spikes: pd.DataFrame, trials: pd.DataFrame, window: tuple[float, float], instruction_n: int = 5,
                        early_window: tuple[int, int] = (5, 15), late_n: int = 10, title: str | None = None) -> plt.Figure:
    """Per-trial rate across the whole session: colour = block context, marker = trial class, per-block early/late means."""
    s = spikes[(spikes["t_rel"] >= window[0]) & (spikes["t_rel"] < window[1])]
    rate = (s.groupby("trial_index").size().reindex(trials["trial_index"], fill_value=0).to_numpy() / (window[1] - window[0]))
    cls = _trial_classes(trials, instruction_n, early_window, late_n)
    x = trials["trial_index"].to_numpy()
    ctx = trials["rewarded_modality"].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 3.4), layout="tight")
    for c, col in CTX_COLOR.items():
        m = ctx == c
        ax.scatter(x[m & (cls == "other")], rate[m & (cls == "other")], s=9, color=col, alpha=0.45, lw=0, label=f"{c} block, other trials")
        ax.scatter(x[m & (cls == "instruction")], rate[m & (cls == "instruction")], s=22, color=col, marker="x", lw=1.2, label=f"{c} block, instruction")
        ax.scatter(x[m & (cls == "early")], rate[m & (cls == "early")], s=26, color=col, edgecolors=EARLYC, lw=1.1, label=f"{c} block, early")
        ax.scatter(x[m & (cls == "late")], rate[m & (cls == "late")], s=26, color=col, marker="s", edgecolors=LATEC, lw=1.1, label=f"{c} block, late")
    ymax = max(rate.max() * 1.15, 1)
    for b, g in trials.groupby("block_index"):
        m = (trials["block_index"] == b).to_numpy()
        x0, x1 = g["trial_index"].min(), g["trial_index"].max()
        ax.axvline(x0 - 0.5, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.axvspan(x0 - 0.5, x0 + instruction_n - 0.5, color=INSTR, alpha=0.15, lw=0)
        ax.text((x0 + x1) / 2, ymax * 0.97, f"block {b} · {ctx[m][0]}", ha="center", va="top", fontsize=8, color=CTX_COLOR[ctx[m][0]])
        for k, colk in (("early", EARLYC), ("late", LATEC)):
            mm = m & (cls == k)
            if mm.any():
                ax.hlines(rate[mm].mean(), x[mm].min() - 0.5, x[mm].max() + 0.5, color=colk, lw=2.2)
    ax.set_ylim(0, ymax); ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_xlabel("trial index (whole session)"); ax.set_ylabel(f"rate in [{window[0]:.2f}, {window[1]:.2f}] s (Hz)")
    ax.legend(fontsize=6.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def context_raster_psth(spikes: pd.DataFrame, trials: pd.DataFrame, window: tuple[float, float], bin_size: float,
                        instruction_n: int = 5, early_window: tuple[int, int] = (5, 15), late_n: int = 10,
                        shade: tuple[float, float] | None = None, include_first_block: bool = False, stim_dur: float = 0.5,
                        title: str | None = None) -> plt.Figure:
    """One raster per context. Rows top-to-bottom: instruction (first 5 trials of each switch block), early
    (trials 5 to 14), late (last 10), each group in chronological order and separated by a line. Below: PSTH per
    context and phase (instruction emphasised)."""
    cls = _trial_classes(trials, instruction_n, early_window, late_n)
    ctx = trials["rewarded_modality"].astype(str).to_numpy()
    keep = (trials["block_index"] > 0).to_numpy() | include_first_block
    spk_groups = {k: v.to_numpy() for k, v in spikes.groupby("trial_index")["t_rel"]}
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    fig = plt.figure(figsize=(8.4, 8.6), layout="tight")
    gs = GridSpec(2, 2, figure=fig, height_ratios=(2.6, 1))
    ax_p = fig.add_subplot(gs[1, :])
    phases = [("instruction", INSTR, "-", 2.0), ("early", EARLYC, "-", 1.2), ("late", LATEC, "--", 1.2)]
    for gi, (c, ccol) in enumerate(CTX_COLOR.items()):
        ax = fig.add_subplot(gs[0, gi])
        row = 0
        ticks, labels = [], []
        for k, kcol, ls, lw in phases:
            tidx = trials.loc[keep & (ctx == c) & (cls == k), "trial_index"].sort_values().to_numpy()
            xs, ys = [], []
            for t in tidx:  # first trial at the top
                sp = spk_groups.get(t)
                if sp is not None:
                    sp = sp[(sp >= window[0]) & (sp < window[1])]
                    xs.append(sp); ys.append(np.full(len(sp), row))
                row += 1
            if xs:
                ax.scatter(np.concatenate(xs), np.concatenate(ys), s=2.2, color=kcol if k != "instruction" else ccol, marker="|", linewidths=0.8)
            if len(tidx):
                ticks.append(row - len(tidx) / 2 - 0.5); labels.append(f"{k}\n(n={len(tidx)})")
                ax.axhline(row - 0.5, color="k", lw=0.6, alpha=0.6)
            all_t = np.concatenate([spk_groups[t] for t in tidx if t in spk_groups]) if len(tidx) else np.array([])
            ax_p.plot(edges[:-1] + bin_size / 2, _psth(all_t, len(tidx), edges), color=ccol, lw=lw, ls=ls,
                      alpha=1.0 if k == "instruction" else 0.75, label=f"{c} {k}")
        if shade:
            ax.axvspan(shade[0], shade[1], color="gold", alpha=0.18, lw=0)
        ax.axvspan(0, stim_dur, color="gray", alpha=0.12, lw=0); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(window); ax.set_ylim(max(row, 1) - 0.5, -0.5)  # inverted: chronological from the top
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(f"{c} blocks (context = rewarded modality)", fontsize=9.5, color=ccol); ax.tick_params(labelsize=8)
        ax.set_xlabel("time from stimulus onset (s)", fontsize=8)
    if shade:
        ax_p.axvspan(shade[0], shade[1], color="gold", alpha=0.18, lw=0, label="analysis window")
    ax_p.axvspan(0, stim_dur, color="gray", alpha=0.12, lw=0); ax_p.axvline(0, color="k", lw=0.6)
    ax_p.set_xlim(window); ax_p.set_xlabel("time from stimulus onset (s)"); ax_p.set_ylabel("rate (Hz)")
    ax_p.legend(fontsize=7, ncol=4, loc="upper right"); ax_p.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def aligned_means_plot(prof_unit: pd.DataFrame, prof_region: pd.DataFrame | None = None, window: str = "baseline",
                       instruction_n: int = 5, early_window: tuple[int, int] = (5, 15), late_n: int = 10,
                       title: str | None = None) -> plt.Figure:
    """Mean rate per within-block position, block-start (left) and block-end (right) aligned, per context."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), layout="tight", width_ratios=(2.3, 1), sharey=True)
    for ax, align in zip(axes, ("block_start", "block_end")):
        for c, col in CTX_COLOR.items():
            p = prof_unit[(prof_unit["context"] == c) & (prof_unit["align"] == align)].sort_values("rel_trial")
            if len(p):
                ax.plot(p["rel_trial"], p["mean_rate"], marker="o", ms=3, lw=1.2, color=col,
                        label=f"{c} blocks (n={int(p['n_blocks'].max())})")
        if prof_region is not None and len(prof_region):
            pr = prof_region[(prof_region["context"] == "both") & (prof_region["align"] == align)].sort_values("rel_trial")
            if len(pr):
                ax.plot(pr["rel_trial"], pr["mean_rate"], color="k", lw=1, alpha=0.5, label=f"{pr['structure'].iloc[0]} rule-updating units, both contexts")
                ax.fill_between(pr["rel_trial"], pr["mean_rate"] - pr["sem_rate"], pr["mean_rate"] + pr["sem_rate"], color="k", alpha=0.08, lw=0)
        if align == "block_start":
            ax.axvspan(-0.5, instruction_n - 0.5, color=INSTR, alpha=0.2, lw=0, label="instruction")
            ax.axvspan(early_window[0] - 0.5, early_window[1] - 0.5, color=EARLYC, alpha=0.12, lw=0, label="early")
            ax.set_xlabel("trial in block (from block start)"); ax.set_ylabel(f"{window} rate (Hz)")
            ax.legend(fontsize=6.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
        else:
            ax.axvspan(-late_n - 0.5, -0.5, color=LATEC, alpha=0.12, lw=0, label="late")
            ax.set_xlabel("trial in block (from block end)")
            ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
        ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    return fig


# ----------------------------------------------------------------------------- tab 2: all-blocks raster, phase PSTH, permutation null
PHASE_COLOR = {"instruction": "#7b2cbf", "early": EARLYC, "late": LATEC, "other": "#b0b0b0"}


def all_blocks_raster_psth(spikes: pd.DataFrame, trials: pd.DataFrame, window: tuple[float, float], bin_size: float,
                           instruction_n: int = 5, early_window: tuple[int, int] = (5, 15), late_n: int = 10,
                           shade: tuple[float, float] | None = None, split_context: bool = False, stim_dur: float = 0.5,
                           title: str | None = None) -> plt.Figure:
    """Top: one raster with every block stacked vertically in session order (block 0 at the top), trials in
    chronological order, plain spike marks, translucent horizontal bands marking the instruction / early / late
    trials of each block, and each block labelled with its rewarded context. Bottom: PSTH of instruction vs early vs late trials pooled over switch blocks
    (optionally split by context)."""
    cls = _trial_classes(trials, instruction_n, early_window, late_n)
    ctx = trials["rewarded_modality"].astype(str).to_numpy()
    blk = trials["block_index"].to_numpy()
    tidx = trials["trial_index"].to_numpy()
    spk_groups = {k: v.to_numpy() for k, v in spikes.groupby("trial_index")["t_rel"]}
    n_tr = len(trials)
    fig = plt.figure(figsize=(8.4, 9.6), layout="tight")
    gs = GridSpec(2, 1, figure=fig, height_ratios=(3.2, 1))
    ax = fig.add_subplot(gs[0]); ax_p = fig.add_subplot(gs[1])
    # raster rows = trial order (already chronological); block 0 at the top via inverted y axis.
    # Spikes are plain (one colour); phase membership is shown by translucent horizontal bands per block.
    xs, ys = [], []
    for row in range(n_tr):
        sp = spk_groups.get(tidx[row])
        if sp is not None:
            sp = sp[(sp >= window[0]) & (sp < window[1])]
            xs.append(sp); ys.append(np.full(len(sp), row))
    if xs:
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=2.0, color="#222222", marker="|", linewidths=0.7)
    band_alpha = {"instruction": 0.28, "early": 0.16, "late": 0.20}
    ticks, labels = [], []
    drawn = set()
    for b in np.unique(blk):
        rows = np.where(blk == b)[0]
        r0, r1 = rows.min(), rows.max()
        c = ctx[rows[0]]
        for k in ("instruction", "early", "late"):
            rk = rows[cls[rows] == k]
            if len(rk):
                ax.axhspan(rk.min() - 0.5, rk.max() + 0.5, color=PHASE_COLOR[k], alpha=band_alpha[k], lw=0,
                           label=f"{k} trials" if k not in drawn else None, zorder=0)
                drawn.add(k)
        ax.axhspan(r0 - 0.5, r1 + 0.5, xmin=0, xmax=0.012, color=CTX_COLOR[c], lw=0)  # context band at the left edge
        ax.axhline(r1 + 0.5, color="k", lw=0.7)
        ticks.append((r0 + r1) / 2); labels.append(f"block {b}\n{'auditory' if c == 'aud' else 'visual'} context")
    if shade:
        ax.axvspan(shade[0], shade[1], color="gold", alpha=0.15, lw=0)
    ax.axvspan(0, stim_dur, color="gray", alpha=0.12, lw=0); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(window); ax.set_ylim(n_tr - 0.5, -0.5)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.5)
    for lab, tk in zip(ax.get_yticklabels(), ticks):
        lab.set_color(CTX_COLOR[ctx[blk == blk[int(round(tk))]][0]])
    ax.tick_params(axis="y", length=0)
    # trial-index ticks on the right (row = trial index, chronological from the top)
    ax_r = ax.secondary_yaxis("right")
    step = 50 if n_tr > 200 else 20
    ax_r.set_yticks(np.arange(0, n_tr, step)); ax_r.tick_params(labelsize=7)
    ax_r.set_ylabel("trial index", fontsize=8)
    ax.set_xlabel("time from stimulus onset (s)", fontsize=8); ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    # PSTH: instruction / early / late over switch blocks
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    sw = blk > 0
    groups = [(k, sw & (cls == k), PHASE_COLOR[k], "-", 1.8 if k == "instruction" else 1.3, None) for k in ("instruction", "early", "late")]
    if split_context:
        groups = [(f"{k} · {c}", sw & (cls == k) & (ctx == c), PHASE_COLOR[k], "-" if c == "aud" else "--", 1.6 if k == "instruction" else 1.1, c)
                  for k in ("instruction", "early", "late") for c in ("aud", "vis")]
    for label, m, col, ls, lw, _ in groups:
        t_sel = tidx[m]
        all_t = np.concatenate([spk_groups[t] for t in t_sel if t in spk_groups]) if len(t_sel) else np.array([])
        ax_p.plot(edges[:-1] + bin_size / 2, _psth(all_t, len(t_sel), edges), color=col, ls=ls, lw=lw, label=f"{label} (n={len(t_sel)})")
    if shade:
        ax_p.axvspan(shade[0], shade[1], color="gold", alpha=0.15, lw=0, label="analysis window")
    ax_p.axvspan(0, stim_dur, color="gray", alpha=0.12, lw=0); ax_p.axvline(0, color="k", lw=0.6)
    ax_p.set_xlim(window); ax_p.set_xlabel("time from stimulus onset (s)"); ax_p.set_ylabel("rate (Hz)")
    ax_p.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False); ax_p.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def kw_null_for_unit(rates: np.ndarray, instruction: np.ndarray, early: np.ndarray, late: np.ndarray, block: np.ndarray,
                     n_perm: int = 5000, seed: int = 0) -> dict:
    """Recompute, for one unit, the Kruskal-Wallis H and Dunn post-hoc mean-rank differences with the within-block
    trial-label-shuffle null. Returns observed values, null arrays and permutation p-values."""
    from scipy.stats import chi2, rankdata
    rng = np.random.default_rng(seed)
    sel = instruction | early | late
    r = rates[sel]; labels = np.where(instruction[sel], 0, np.where(early[sel], 1, 2)); blk = block[sel]
    N = len(r); ranks = rankdata(r); n_g = np.bincount(labels, minlength=3).astype(float)
    _, t = np.unique(r, return_counts=True); C = 1 - ((t.astype(float) ** 3 - t).sum() / (N ** 3 - N))

    def stats_of(lab):
        S = np.array([ranks[lab == g].sum() for g in range(3)])
        H = 12.0 / (N * (N + 1)) * (S ** 2 / n_g).sum() - 3 * (N + 1)
        mr = S / n_g
        return (H / C if C > 0 else np.nan), mr[0] - mr[1], mr[0] - mr[2]
    H_obs, de_obs, dl_obs = stats_of(labels)
    Hn, den, dln = np.empty(n_perm), np.empty(n_perm), np.empty(n_perm)
    blocks = np.unique(blk)
    for k in range(n_perm):
        perm = labels.copy()
        for b in blocks:
            m = blk == b
            perm[m] = rng.permutation(labels[m])
        Hn[k], den[k], dln[k] = stats_of(perm)
    return {"H_obs": H_obs, "H_null": Hn, "p_perm": (1 + (Hn >= H_obs).sum()) / (n_perm + 1), "p_chi2": float(chi2.sf(H_obs, 2)) if np.isfinite(H_obs) else 1.0,
            "d_early_obs": de_obs, "d_early_null": den, "p_early": (1 + (den >= de_obs).sum()) / (n_perm + 1),
            "d_late_obs": dl_obs, "d_late_null": dln, "p_late": (1 + (dln >= dl_obs).sum()) / (n_perm + 1),
            "n": {"instruction": int(n_g[0]), "early": int(n_g[1]), "late": int(n_g[2])}, "n_perm": n_perm}


def kw_null_plot(res: dict, title: str | None = None) -> plt.Figure:
    """Three histograms: null of Kruskal-Wallis H and of the two post-hoc mean-rank differences, observed values marked."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), layout="tight")
    panels = [("Kruskal-Wallis H", res["H_null"], res["H_obs"], res["p_perm"]),
              ("rank diff: instr − early", res["d_early_null"], res["d_early_obs"], res["p_early"]),
              ("rank diff: instr − late", res["d_late_null"], res["d_late_obs"], res["p_late"])]
    for ax, (name, null, obs, p) in zip(axes, panels):
        ax.hist(null, bins=40, color="#9aa5b1", alpha=0.9)
        q95 = np.nanpercentile(null, 95)
        ax.axvline(q95, color="k", lw=0.8, ls=":", label="null 95th pct")
        ax.axvline(obs, color=PHASE_COLOR["instruction"], lw=2, label=f"observed (p = {p:.3g})")
        ax.set_title(name, fontsize=8); ax.tick_params(labelsize=7); ax.set_yticks([])
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=False, handlelength=1.2)
    axes[0].set_ylabel(f"{res['n_perm']} within-block label shuffles", fontsize=7.5)
    if title:
        fig.suptitle(title, fontsize=9)
    return fig
