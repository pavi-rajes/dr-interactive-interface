#!/usr/bin/env python3
"""Smoke test of the app's data layer and plot helpers (no Shiny session needed)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_data as D  # noqa: E402
import app_plots as PL  # noqa: E402
from brain3d import brain_figure  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work/shiny-app-tabs/data")
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
ctx = D.context_table(); rule = D.rule_table()
best = ctx[ctx.is_robust].sort_values("p_perm").iloc[0]
sid, ui = best.session_id, int(best.unit_index)
tr = D.load_trials(sid); sp = D.unit_spikes(sid, ui)
print(f"unit {sid}/{ui}: {len(sp)} aligned spikes, loaded in {time.time() - t0:.2f}s")
aud = tr.loc[tr.rewarded_modality == "aud", "trial_index"].to_numpy(); vis = tr.loc[tr.rewarded_modality == "vis", "trial_index"].to_numpy()
PL.raster_psth(sp, tr, [("auditory blocks", aud, PL.AUD), ("visual blocks", vis, PL.VIS)], (-1.5, 2.0), 0.05, shade=(-1.5, -0.06),
               title=f"{sid} unit {ui}").savefig(out / "plots_smoke_raster_context.png", dpi=110)
PL.per_trial_rate_plot(sp, tr, (-1.5, -0.06)).savefig(out / "plots_smoke_trial_rates.png", dpi=110)
both = rule[(rule.direction == "both") & rule.is_rule_updating].sort_values("bl_p_perm").iloc[0]
sid2, ui2 = both.session_id, int(both.unit_index)
tr2 = D.load_trials(sid2); sp2 = D.unit_spikes(sid2, ui2)
sw = tr2.block_index > 0; early = sw & (tr2.trial_index_in_block < 15); late = sw & (tr2.trial_index_in_block >= 30)
a, v = tr2.rewarded_modality == "aud", tr2.rewarded_modality == "vis"
PL.raster_psth(sp2, tr2, [("aud early", tr2.loc[early & a, "trial_index"].to_numpy(), PL.AUD), ("aud late", tr2.loc[late & a, "trial_index"].to_numpy(), "#f2b27a"),
                          ("vis early", tr2.loc[early & v, "trial_index"].to_numpy(), PL.VIS), ("vis late", tr2.loc[late & v, "trial_index"].to_numpy(), "#8fbce6")],
               (-1.5, 2.0), 0.05, shade=(-1.5, -0.06), sort_by="trial_index_in_block", title=f"{sid2} unit {ui2}").savefig(out / "plots_smoke_raster_rule.png", dpi=110)
PL.switch_profile_plot(D.unit_profile(sid2, ui2, "baseline"), D.structure_profile(both.structure, "baseline")).savefig(out / "plots_smoke_profile.png", dpi=110)
f1 = brain_figure(ctx[ctx.is_robust], color_col="log2_ratio", region_meshes=["MOs"], selected_unit=(sid, ui))
f2 = brain_figure(rule[(rule.direction == "both") & rule.is_rule_updating], color_col="bl_log2_ratio", selected_unit=(sid2, ui2))
print("brain figures:", len(f1.data), len(f2.data), "traces; total", f"{time.time() - t0:.1f}s")
print("PASS smoke")
