"""Dynamic Routing interactive viewer (Python Shiny).

Tab 1: context-dependent baseline shifts. Tab 2: rule-change units (instruction / early / late
trials within blocks, organised by block context = rewarded modality).
Reads only precomputed artifacts from config.PRECOMPUTED_DIR (see code/README.md).
Run: shiny run code/app.py --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from shiny import App, reactive, render, req, ui  # noqa: E402
from shinywidgets import output_widget, render_widget  # noqa: E402

import app_data as D  # noqa: E402
import app_plots as PL  # noqa: E402
from brain3d import brain_figure, load_mesh_index, to_plot_coords  # noqa: E402

SESSIONS = D.sessions()
MESH_STRUCTS = sorted(load_mesh_index()["structures"].keys() - {"root"})
CTX = D.context_table()
RULE = D.rule_table()
SUMM = D.summaries()
RP = SUMM["rule"]["params"]
INSTR_N, EARLY_WIN, LATE_N = RP["instruction_n"], tuple(RP["early_window"]), RP["late_n"]
DEFAULT_TAB = os.environ.get("DR_APP_DEFAULT_TAB", "tab_context")

# tab 1 dot tiers: label -> flag column (None = all units)
C_TIERS = {"is_robust": "robust: BH q < 0.05, block-consistent, rate >= 0.1 Hz", "is_significant": "significant: BH q < 0.05 only", "all": "all QC-pass units"}
# tab 2 contrasts and tiers
R_CONTRASTS = {"kw_": f"instruction (trials 0 to {INSTR_N - 1}) > early and late: Kruskal-Wallis + post-hoc, trial-shuffle null",
               "": f"early (trials {EARLY_WIN[0]} to {EARLY_WIN[1] - 1}) vs late (last {LATE_N}): chunk permutation",
               "inst_": f"instruction vs late (last {LATE_N}): chunk permutation"}
R_TIERS = {
    "kw_": {"is_instruction_higher": "rule-updating (definition): instruction rate > early and > late in the selected window",
            "is_rule_updating": "stricter: also Kruskal-Wallis omnibus p < 0.05 and post-hoc p < 0.05, rate >= 0.1 Hz",
            "is_candidate": "stricter: the same at p < 0.01", "is_bh": "supplementary: BH-corrected version (BH on hold)", "all": "all QC-pass units"},
    "": {"is_candidate": "candidate: p < 0.01 (uncorrected), block-consistent, rate >= 0.1 Hz",
         "is_robust": "robust: BH q < 0.05, block-consistent, rate >= 0.1 Hz",
         "is_liberal": "liberal: p < 0.05 (uncorrected), block-consistent, rate >= 0.1 Hz", "all": "all QC-pass units"},
}
R_TIERS["inst_"] = R_TIERS[""]
R_CONTEXTS = {"both": "both (all 5 switch blocks)", "aud": "auditory blocks", "vis": "visual blocks"}


def fmt(x, nd=3):
    try:
        return "nan" if pd.isna(x) else (f"{x:.{nd}g}" if isinstance(x, (float, np.floating)) else str(x))
    except TypeError:
        return str(x)


def stats_table(rows: list[tuple[str, str]]):
    return ui.tags.table({"class": "table table-sm mb-0", "style": "font-size:0.85rem"},
                         ui.tags.tbody(*[ui.tags.tr(ui.tags.th(k, style="width:48%"), ui.tags.td(v)) for k, v in rows]))


def tier_col(sel: str) -> str | None:
    return None if sel == "all" else sel


def common_controls(p: str, default_shade: tuple[float, float], region_default: list[str]):
    return [
        ui.input_select(f"{p}_session", "Session", choices=SESSIONS, selected=SESSIONS[-1]),
        ui.input_select(f"{p}_structure", "Brain region (flagged / all units)", choices=[]),
        ui.input_select(f"{p}_unit", "Unit", choices=[]),
        ui.input_slider(f"{p}_analysis", "Analysis window (s from stim onset)", min=-2.0, max=3.0, value=default_shade, step=0.01),
        ui.input_slider(f"{p}_plot_window", "Raster / PSTH window (s)", min=-2.0, max=3.0, value=(-1.5, 2.0), step=0.1),
        ui.input_slider(f"{p}_bin", "PSTH bin (s)", min=0.01, max=0.25, value=0.05, step=0.01),
        ui.input_selectize(f"{p}_regions", "Region meshes on the brain", choices=MESH_STRUCTS, selected=region_default, multiple=True),
        ui.input_checkbox(f"{p}_all_sessions", "Brain dots from all sessions", True),
    ]


tab_context = ui.nav_panel(
    "Context baseline shifts",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("c_tier", "Units shown as dots / in the list", choices=C_TIERS, selected="is_robust"),
            *common_controls("c", (-1.5, -0.06), ["MOs", "CP", "AUDp"]),
            ui.input_select("c_color", "Colour dots by", choices={"log2_ratio": "log2(aud / vis) baseline rate",
                                                                    "modulation_index": "modulation index (aud - vis)/(aud + vis)"}),
            width=320, open="always"),
        ui.layout_columns(
            ui.div(
                ui.card(ui.card_header("Rasters and PSTH: auditory vs visual blocks"), ui.output_plot("c_raster", height="560px")),
                ui.card(ui.card_header("Per-trial rate in the analysis window across the session"), ui.output_plot("c_trial_rates", height="270px")),
            ),
            ui.div(
                ui.card(ui.card_header("Units with context-dependent baseline shifts (click a dot to select)"), output_widget("brain_context", height="560px")),
                ui.card(ui.card_header("Selected unit"), ui.output_ui("c_stats")),
                ui.card(ui.card_header("Regions (>= 20 units)"), ui.output_data_frame("c_region_table")),
            ),
            col_widths=(7, 5)),
    ),
    value="tab_context")

tab_rule = ui.nav_panel(
    "Rule-change units",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_radio_buttons("r_context", "Context (rewarded modality of the block)", choices=R_CONTEXTS, selected="both"),
            ui.input_radio_buttons("r_contrast", "Test", choices=R_CONTRASTS, selected="kw_"),
            ui.input_select("r_tier", "Units shown as dots / in the list", choices=R_TIERS["kw_"], selected="is_instruction_higher"),
            ui.input_radio_buttons("r_window", "Statistics window", choices={"baseline": "baseline (-1.5 to -0.06 s)", "evoked": "evoked (0 to 0.5 s)"},
                                   selected="baseline", inline=True),
            *common_controls("r", (-1.5, -0.06), ["MOs", "CP", "PL"]),
            ui.input_checkbox("r_split_ctx", "Split PSTH by context", False),
            ui.input_checkbox("r_pop_region", "Population dynamics: selected region only", False),
            ui.input_checkbox("r_first_block", "Include block 0 in the per-context view (no preceding rule)", False),
            width=320, open="always"),
        ui.layout_columns(
            ui.div(
                ui.card(ui.card_header("Block timeline: per-trial rate, context, instruction / early / late trials"), ui.output_plot("r_timeline", height="330px")),
                ui.card(ui.card_header("Raster: all blocks stacked in session order (shaded bands = instruction, early, late trials; rewarded context labelled) and PSTH: instruction vs early vs late"),
                        ui.output_plot("r_raster", height="860px")),
                ui.card(ui.card_header("Mean rate by position in block, per context"), ui.output_plot("r_profile", height="290px")),
                ui.card(ui.card_header("Instruction-trial dynamics: selected unit and the displayed population (trials 0 to 14 of switch blocks)"),
                        ui.output_plot("r_dynamics", height="330px")),
            ),
            ui.div(
                ui.card(ui.card_header("Rule-change units (click a dot to select)"), output_widget("brain_rule", height="560px")),
                ui.card(ui.card_header("Selected unit: permutation null for the current analysis window and context"),
                        ui.output_plot("r_null_fig", height="250px"), ui.output_ui("r_null_stats")),
                ui.card(ui.card_header("Selected unit: stored statistics (fixed windows)"), ui.output_ui("r_stats")),
                ui.card(ui.card_header("Regions (>= 20 units)"), ui.output_data_frame("r_region_table")),
            ),
            col_widths=(7, 5)),
    ),
    value="tab_rule")

app_ui = ui.page_navbar(tab_context, tab_rule, title="Dynamic Routing neural viewer", id="tabs", selected=DEFAULT_TAB, fillable=False)


def server(input, output, session):
    # ------------------------------------------------------------------ shared selection wiring
    def wire_selection(p: str, table_fn, flag_col_fn, unit_label_fn):
        """Keep structure/unit selects in sync with session / tier; returns (pending Value, key calc).
        table_fn() -> DataFrame with one row per unit for the current settings; flag_col_fn() -> column name or None."""
        pending = reactive.Value(None)

        @reactive.calc
        def table_session():
            t = table_fn()
            return t[t["session_id"] == input[f"{p}_session"]()]

        @reactive.effect
        def _structures():
            sid = input[f"{p}_session"]()
            flag = flag_col_fn()
            g = D.structures_for(table_fn(), sid, flag, only_flagged=flag is not None)
            choices = {r.structure: f"{r.structure}  ({r.n_flagged} / {r.n_units})" for r in g.itertuples()}
            with reactive.isolate():
                cur = input[f"{p}_structure"]()
                pend = pending.get()
            sel = None
            if pend and pend[0] == sid:
                t = table_fn()
                m = (t["session_id"] == sid) & (t["unit_index"] == pend[1])
                if m.any():
                    sel = t.loc[m, "structure"].iloc[0]
            if sel not in choices:
                sel = cur if cur in choices else (next(iter(choices)) if choices else None)
            ui.update_select(f"{p}_structure", choices=choices, selected=sel)

        @reactive.effect
        def _units():
            sid = input[f"{p}_session"]()
            struct = input[f"{p}_structure"]()
            flag = flag_col_fn()
            t = table_session()
            t = t[t["structure"] == struct]
            if flag is not None:
                t = t[t[flag]]
            t = unit_label_fn(t)
            choices = {str(int(r.unit_index)): r.label for r in t.itertuples()}
            with reactive.isolate():
                cur = input[f"{p}_unit"]()
                pend = pending.get()
            sel = None
            if pend and pend[0] == sid and str(pend[1]) in choices:
                sel = str(pend[1])
                pending.set(None)
            if sel is None:
                sel = cur if cur in choices else (next(iter(choices)) if choices else None)
            ui.update_select(f"{p}_unit", choices=choices, selected=sel)

        @reactive.calc
        def key():
            sid = input[f"{p}_session"]()
            u = input[f"{p}_unit"]()
            req(u)
            return sid, int(u)

        return pending, key

    def make_brain(fig: go.Figure, p: str, pending: reactive.Value) -> go.FigureWidget:
        fw = go.FigureWidget(fig)

        def handler(trace, points, selector):
            if not points.point_inds:
                return
            cd = trace.customdata[points.point_inds[0]]
            pending.set((str(cd[0]), int(cd[1])))
            ui.update_select(f"{p}_session", selected=str(cd[0]))

        for tr in fw.data:
            if tr.name and tr.name.startswith("units"):
                tr.on_click(handler)
        return fw

    def move_highlight(widget_out, tab: pd.DataFrame, sid: str, ui_: int):
        fw = widget_out.widget if hasattr(widget_out, "widget") else None
        if fw is None:
            return
        m = (tab["session_id"] == sid) & (tab["unit_index"] == ui_)
        if not m.any():
            return
        x, y, z = to_plot_coords(tab.loc[m, "ccf_ap"], tab.loc[m, "ccf_dv"], tab.loc[m, "ccf_ml"])
        for tr in fw.data:
            if tr.name == "selected unit":
                tr.x, tr.y, tr.z = list(x), list(y), list(z)

    # ------------------------------------------------------------------ tab 1: context shifts
    def c_label(t: pd.DataFrame) -> pd.DataFrame:
        t = t.assign(alr=t["log2_ratio"].abs()).sort_values(["p_perm", "alr"], ascending=[True, False])
        t["label"] = [f"{int(r.unit_index)} | log2 {r.log2_ratio:+.2f} | q {r.q_bh:.2g}{'' if r.is_robust else ' (n.s.)'}" for r in t.itertuples()]
        return t

    c_pending, c_key = wire_selection("c", D.context_table, lambda: tier_col(input.c_tier()), c_label)

    @reactive.calc
    def c_row():
        sid, ui_ = c_key()
        m = (CTX["session_id"] == sid) & (CTX["unit_index"] == ui_)
        req(m.any())
        return CTX[m].iloc[0]

    @reactive.calc
    def c_spikes():
        sid, ui_ = c_key()
        return D.unit_spikes(sid, ui_)

    @render.ui
    def c_stats():
        r = c_row()
        return stats_table([
            ("unit", f"{r['unit_id']}  ({r['structure']}, {r['location']})"),
            ("CCF ap / dv / ml (µm)", f"{fmt(r['ccf_ap'], 5)} / {fmt(r['ccf_dv'], 5)} / {fmt(r['ccf_ml'], 5)}"),
            ("baseline rate aud / vis (Hz)", f"{r['mean_rate_aud']:.2f} / {r['mean_rate_vis']:.2f}"),
            ("log2(aud / vis), modulation index", f"{r['log2_ratio']:+.2f}, {fmt(r['modulation_index'])}"),
            ("p (chunk permutation), q (BH)", f"{r['p_perm']:.2g}, {r['q_bh']:.2g}"),
            ("block consistency, full separation", f"{fmt(r['block_consistency'])}, {r['full_block_separation']}"),
            ("flags", f"robust = {r['is_robust']}, significant = {r['is_significant']}, direction = {r['shift_direction']}"),
            ("task firing rate (Hz)", f"{r['task_firing_rate']:.2f}"),
        ])

    @render.plot
    def c_raster():
        sid, ui_ = c_key()
        tr = D.load_trials(sid)
        aud = tr.loc[tr["rewarded_modality"] == "aud", "trial_index"].to_numpy()
        vis = tr.loc[tr["rewarded_modality"] == "vis", "trial_index"].to_numpy()
        return PL.raster_psth(c_spikes(), tr, [("auditory blocks", aud, PL.AUD), ("visual blocks", vis, PL.VIS)],
                              window=tuple(input.c_plot_window()), bin_size=input.c_bin(), shade=tuple(input.c_analysis()),
                              title=f"{sid} unit {ui_} ({c_row()['structure']})")

    @render.plot
    def c_trial_rates():
        sid, ui_ = c_key()
        return PL.per_trial_rate_plot(c_spikes(), D.load_trials(sid), tuple(input.c_analysis()))

    @render_widget
    def brain_context():
        flag = tier_col(input.c_tier())
        t = CTX[CTX[flag]] if flag else CTX
        if not input.c_all_sessions():
            t = t[t["session_id"] == input.c_session()]
        with reactive.isolate():
            try:
                sel = c_key()
            except Exception:  # noqa: BLE001
                sel = None
        col = input.c_color()
        fig = brain_figure(t, color_col=col, color_range=(-3, 3) if col == "log2_ratio" else (-1, 1), region_meshes=list(input.c_regions()),
                           selected_unit=sel, colorbar_title="log2 aud/vis" if col == "log2_ratio" else "mod. index",
                           hover_cols=("unit_id", "structure", "session_id", "shift_direction"))
        fig.update_layout(height=540)
        return make_brain(fig, "c", c_pending)

    @reactive.effect
    def _c_highlight():
        try:
            sid, ui_ = c_key()
        except Exception:  # noqa: BLE001
            return
        move_highlight(brain_context, CTX, sid, ui_)

    @render.data_frame
    def c_region_table():
        r = D.context_regions()
        r = r[r["n_units"] >= 20][["structure", "n_units", "n_robust", "frac_robust", "n_aud_higher_robust", "n_vis_higher_robust", "n_sessions"]]
        return render.DataGrid(r.round(3), height="240px", summary=False)

    # ------------------------------------------------------------------ tab 2: rule change
    def r_w() -> str:
        return "ev" if input.r_window() == "evoked" else "bl"

    def r_c() -> str:
        return input.r_contrast()

    @reactive.effect
    @reactive.event(input.r_contrast)
    def _r_tier_choices():
        c = input.r_contrast()
        with reactive.isolate():
            cur = input.r_tier()
        choices = R_TIERS[c]
        ui.update_select("r_tier", choices=choices, selected=cur if cur in choices else next(iter(choices)))

    def r_flag() -> str | None:
        """Flag column for the selected window x test x tier, e.g. bl_kw_is_instruction_higher or ev_inst_is_candidate."""
        tier = input.r_tier()
        if tier == "all":
            return None
        col = f"{r_w()}_{r_c()}{tier}"
        return col if col in RULE.columns else None

    def r_table():
        return RULE[RULE["context"] == input.r_context()]

    def r_label(t: pd.DataFrame) -> pd.DataFrame:
        w, c = r_w(), r_c()
        t = t.assign(alr=t[f"{w}_{c}log2_ratio"].abs()).sort_values([f"{w}_{c}p_perm", "alr"], ascending=[True, False])
        first = {"kw_": "instr/rest", "": "early/late", "inst_": "instr/late"}[c]
        t["label"] = [f"{int(r.unit_index)} | log2 {first} {getattr(r, f'{w}_{c}log2_ratio'):+.2f} | p {getattr(r, f'{w}_{c}p_perm'):.2g} | q {getattr(r, f'{w}_{c}q_bh'):.2g}"
                      for r in t.itertuples()]
        return t

    r_pending, r_key = wire_selection("r", r_table, r_flag, r_label)

    @reactive.effect
    @reactive.event(input.r_window)
    def _r_window_default():
        ui.update_slider("r_analysis", value=(0.0, 0.5) if input.r_window() == "evoked" else (-1.5, -0.06))

    @reactive.calc
    def r_row():
        sid, ui_ = r_key()
        t = r_table()
        m = (t["session_id"] == sid) & (t["unit_index"] == ui_)
        req(m.any())
        return t[m].iloc[0]

    @reactive.calc
    def r_spikes():
        sid, ui_ = r_key()
        return D.unit_spikes(sid, ui_)

    @render.ui
    def r_stats():
        r = r_row()
        w = r_w()
        wn = input.r_window()

        def contrast_rows(c: str, first: str):
            return [
                (f"{wn} rate {first} / late (Hz)", f"{r[f'{w}_mean_{first}']:.2f} / {r[f'{w}_mean_late']:.2f}"),
                (f"log2({first} / late), sign", f"{r[f'{w}_{c}log2_ratio']:+.2f}, {r[f'{w}_{c}direction_sign']}"),
                ("p (chunk permutation), q (BH)", f"{r[f'{w}_{c}p_perm']:.2g}, {r[f'{w}_{c}q_bh']:.2g}" + (f" (raw evoked p {r[f'ev_{c}p_perm_raw']:.2g})" if w == "ev" else "")),
                ("block consistency", fmt(r[f"{w}_{c}consistency"])),
                ("tiers", f"robust = {r[f'{w}_{c}is_robust']}, candidate = {r[f'{w}_{c}is_candidate']}, liberal = {r[f'{w}_{c}is_liberal']}"),
            ]
        kw_rows = [
            (f"{wn} rate instruction / early / late (Hz)", f"{r[f'{w}_mean_instruction']:.2f} / {r[f'{w}_mean_early']:.2f} / {r[f'{w}_mean_late']:.2f}"),
            ("Kruskal-Wallis H; omnibus p (trial-shuffle permutation)", f"{r[f'{w}_kw_h']:.2f}; {r[f'{w}_kw_p_perm']:.2g}" + (f" (stimulus-residualized p {r['ev_kw_p_perm_residualized']:.2g})" if w == "ev" else "")),
            ("post-hoc instruction > early: mean-rank diff, p", f"{r[f'{w}_kw_d_early']:+.1f}, {r[f'{w}_kw_p_early']:.2g}"),
            ("post-hoc instruction > late: mean-rank diff, p", f"{r[f'{w}_kw_d_late']:+.1f}, {r[f'{w}_kw_p_late']:.2g}"),
            ("log2(instruction / rest)", f"{r[f'{w}_kw_log2_ratio']:+.2f}"),
            ("rule-updating (definition: instruction > early and > late)", f"{r[f'{w}_kw_is_instruction_higher']} (windows: {r['instruction_higher_window']})"),
            ("stricter tiers", f"KW + post-hoc p<0.05 = {r[f'{w}_kw_is_rule_updating']}, p<0.01 = {r[f'{w}_kw_is_candidate']}"),
            ("supplementary BH (on hold): omnibus q, post-hoc q early / late", f"{r[f'{w}_kw_q_bh']:.2g}, {r[f'{w}_kw_q_unit_early']:.2g} / {r[f'{w}_kw_q_unit_late']:.2g}"),
        ]
        return ui.div(
            stats_table([
                ("unit", f"{r['unit_id']}  ({r['structure']}, {r['location']})"),
                ("CCF ap / dv / ml (µm)", f"{fmt(r['ccf_ap'], 5)} / {fmt(r['ccf_dv'], 5)} / {fmt(r['ccf_ml'], 5)}"),
                ("context, switch blocks", f"{r['context']} ({r['direction']}), {int(r['n_blocks'])} blocks"),
            ]),
            ui.h6("Instruction vs early and late (Kruskal-Wallis + post-hoc, primary)", class_="mt-2 mb-1"), stats_table(kw_rows),
            ui.h6("Early vs late (chunk permutation)", class_="mt-2 mb-1"), stats_table(contrast_rows("", "early")),
            ui.h6("Instruction vs late (chunk permutation)", class_="mt-2 mb-1"), stats_table(contrast_rows("inst_", "instruction")),
        )

    @render.plot
    def r_timeline():
        sid, ui_ = r_key()
        return PL.block_timeline_plot(r_spikes(), D.load_trials(sid), tuple(input.r_analysis()), INSTR_N, EARLY_WIN, LATE_N,
                                      title=f"{sid} unit {ui_} ({r_row()['structure']})")

    @render.plot
    def r_raster():
        sid, ui_ = r_key()
        return PL.all_blocks_raster_psth(r_spikes(), D.load_trials(sid), tuple(input.r_plot_window()), input.r_bin(), INSTR_N, EARLY_WIN, LATE_N,
                                         shade=tuple(input.r_analysis()), split_context=input.r_split_ctx(),
                                         title=f"{sid} unit {ui_} ({r_row()['structure']})")

    @reactive.calc
    def r_null_calc():
        """Live Kruskal-Wallis + post-hoc null for the selected unit, analysis window and context (5,000 shuffles)."""
        sid, ui_ = r_key()
        tr = D.load_trials(sid)
        sp = r_spikes()
        w0, w1 = input.r_analysis()
        cnt = sp[(sp["t_rel"] >= w0) & (sp["t_rel"] < w1)].groupby("trial_index").size().reindex(tr["trial_index"], fill_value=0).to_numpy()
        rates = cnt / (w1 - w0)
        blk = tr["block_index"].to_numpy(); tib = tr["trial_index_in_block"].to_numpy()
        nb = tr.groupby("block_index")["trial_index"].transform("size").to_numpy()
        in_ctx = (blk > 0) & ((tr["rewarded_modality"].astype(str).to_numpy() == input.r_context()) if input.r_context() != "both" else True)
        inst = in_ctx & (tib < INSTR_N); early = in_ctx & (tib >= EARLY_WIN[0]) & (tib < EARLY_WIN[1]); late = in_ctx & (tib >= nb - LATE_N)
        res = PL.kw_null_for_unit(rates, inst, early, late, blk, n_perm=5000, seed=0)
        res["means"] = {"instruction": float(rates[inst].mean()), "early": float(rates[early].mean()), "late": float(rates[late].mean())}
        return res

    @render.plot
    def r_null_fig():
        res = r_null_calc()
        m = res["means"]
        return PL.kw_null_plot(res, title=f"instr {m['instruction']:.2f} / early {m['early']:.2f} / late {m['late']:.2f} Hz  ·  "
                                          f"window {input.r_analysis()[0]:.2f} to {input.r_analysis()[1]:.2f} s  ·  context {input.r_context()}")

    @render.ui
    def r_null_stats():
        res = r_null_calc()
        n = res["n"]
        return stats_table([
            ("trials: instruction / early / late", f"{n['instruction']} / {n['early']} / {n['late']}"),
            ("Kruskal-Wallis H observed; null mean, 95th, 99th pct", f"{res['H_obs']:.2f}; {np.nanmean(res['H_null']):.2f}, {np.nanpercentile(res['H_null'], 95):.2f}, {np.nanpercentile(res['H_null'], 99):.2f}"),
            ("omnibus p (permutation), p (chi2, df 2)", f"{res['p_perm']:.3g}, {res['p_chi2']:.3g}"),
            ("post-hoc instruction > early: mean-rank diff, null 95th pct, p", f"{res['d_early_obs']:+.1f}, {np.nanpercentile(res['d_early_null'], 95):+.1f}, {res['p_early']:.3g}"),
            ("post-hoc instruction > late: mean-rank diff, null 95th pct, p", f"{res['d_late_obs']:+.1f}, {np.nanpercentile(res['d_late_null'], 95):+.1f}, {res['p_late']:.3g}"),
            ("definition met (instruction > early and > late)", str(res["means"]["instruction"] > res["means"]["early"] and res["means"]["instruction"] > res["means"]["late"])),
        ])

    @render.plot
    def r_profile():
        sid, ui_ = r_key()
        w = input.r_window()
        return PL.aligned_means_plot(D.unit_profile(sid, ui_, w), D.structure_profile(r_row()["structure"], w), window=w,
                                     instruction_n=INSTR_N, early_window=EARLY_WIN, late_n=LATE_N)

    @reactive.calc
    def r_population_profiles():
        """Fold-change profiles (block_start positions 0..14, pooled contexts) for the displayed rule-updating set."""
        t = r_table()
        flag = r_flag()
        if flag:
            t = t[t[flag]]
        if not input.r_all_sessions():
            t = t[t["session_id"] == input.r_session()]
        if input.r_pop_region():
            t = t[t["structure"] == input.r_structure()]
        if t.empty:
            return pd.DataFrame(columns=["unit_key", "rel_trial", "fold"])
        keys = t[["session_id", "unit_index"]].drop_duplicates()
        prof = D.rule_profiles()
        prof = prof[(prof["context"] == "both") & (prof["window"] == input.r_window())].merge(keys, on=["session_id", "unit_index"])
        late = prof[(prof["align"] == "block_end") & (prof["rel_trial"] >= -LATE_N)].groupby(["session_id", "unit_index"])["mean_rate"].mean().rename("late")
        start = prof[(prof["align"] == "block_start") & (prof["rel_trial"] < 15)].merge(late.reset_index(), on=["session_id", "unit_index"])
        start["fold"] = (start["mean_rate"] + 0.1) / (start["late"] + 0.1)
        start["unit_key"] = start["session_id"] + ":" + start["unit_index"].astype(str)
        return start[["unit_key", "rel_trial", "fold"]]

    @render.plot
    def r_dynamics():
        sid, ui_ = r_key()
        w = input.r_window()
        return PL.instruction_dynamics_plot(D.unit_profile(sid, ui_, w), r_population_profiles(), n_pos=15, instruction_n=INSTR_N, window=w,
                                            title=f"{sid} unit {ui_} ({r_row()['structure']}); population = current dot set, {w} window")

    @render_widget
    def brain_rule():
        t = r_table()
        flag = r_flag()
        if flag:
            t = t[t[flag]]
        if not input.r_all_sessions():
            t = t[t["session_id"] == input.r_session()]
        with reactive.isolate():
            try:
                sel = r_key()
            except Exception:  # noqa: BLE001
                sel = None
        w, c = r_w(), r_c()
        cb = {"kw_": "log2 instr/rest", "": "log2 early/late", "inst_": "log2 instr/late"}[c]
        fig = brain_figure(t, color_col=f"{w}_{c}log2_ratio", color_range=(-3, 3), region_meshes=list(input.r_regions()), selected_unit=sel,
                           colorbar_title=cb, hover_cols=("unit_id", "structure", "session_id", f"{w}_{c}direction_sign"))
        fig.update_layout(height=540)
        return make_brain(fig, "r", r_pending)

    @reactive.effect
    def _r_highlight():
        try:
            sid, ui_ = r_key()
        except Exception:  # noqa: BLE001
            return
        move_highlight(brain_rule, RULE[RULE["context"] == "both"], sid, ui_)

    @render.data_frame
    def r_region_table():
        """Per brain region: units recorded (incl. QC-fail), QC-pass units, rule-updating units for the selected window /
        test / tier, units expected under the label-shuffle null, and mean rate in the fixed statistics window for
        instruction / early / late trials over the rule-updating units."""
        t = r_table()
        w, c = r_w(), r_c()
        flag = r_flag()
        rec = D.recorded_units_by_structure()
        rows = []
        null_col = {"is_instruction_higher": f"{w}_kw_null_rate_def", "is_rule_updating": f"{w}_kw_null_rate_p05",
                    "is_candidate": f"{w}_kw_null_rate_p01"}.get(input.r_tier()) if c == "kw_" else None
        for struct, g in t.groupby("structure"):
            f = g[g[flag]] if flag else g
            rows.append({
                "region": struct,
                "units recorded": int(rec.get(struct, {}).get("n_units", 0)),
                "units passing QC": int(len(g)),
                "rule-updating": int(len(f)),
                "expected under null": round(float(g[null_col].sum()), 1) if null_col and null_col in g.columns else np.nan,
                f"instruction rate (Hz, {input.r_window()})": round(float(f[f"{w}_mean_instruction"].mean()), 2) if len(f) else np.nan,
                "early rate (Hz)": round(float(f[f"{w}_mean_early"].mean()), 2) if len(f) else np.nan,
                "late rate (Hz)": round(float(f[f"{w}_mean_late"].mean()), 2) if len(f) else np.nan,
                "sessions": int(g["session_id"].nunique()),
            })
        r = pd.DataFrame(rows)
        r = r[r["units passing QC"] >= 20].sort_values(["rule-updating", "units passing QC"], ascending=[False, False]).reset_index(drop=True)
        return render.DataGrid(r, height="320px", summary=False)


app = App(app_ui, server)
