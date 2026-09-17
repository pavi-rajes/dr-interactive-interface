---
slug: ccf-brain-3d-component
status: done
plan_review_attempts: 2
---

# Goal
Provide the 3D mouse brain used on the right side of both tabs. Obtain the Allen CCF (CCFv3, 25 or 50 µm) root brain mesh, e.g. via brainglobe-atlasapi, the Allen Institute mesh download, or allensdk, and save a decimated version to `data/precomputed/ccf_root_mesh.npz` (vertices and faces) via a reproducible script `code/fetch_ccf_mesh.py`. Optionally also fetch a few region meshes (e.g. VISp, AUDp, MOs, CP) for context.

Build a reusable Python function (e.g. `code/brain3d.py`) that returns a plotly `Figure` with the translucent brain mesh and a scatter3d of unit dots given a units dataframe with ccf_ap/ccf_dv/ccf_ml, color-coded by a chosen column (e.g. significance or effect size) with hover text (unit id, structure, session), and that supports highlighting a selected unit. Verify that unit CCF coordinates (in µm) align with the mesh coordinate frame and document the axis convention. Include a small standalone demo that writes an HTML file for visual inspection.

# Instructions

Paths relative to `/Users/pavir/Code/dr-interactive-interface`; interpreter `/opt/miniconda3/bin/python3` (plotly 6.7, numpy, pandas, brainglobe-atlasapi 2.3.1 installed; the `allen_mouse_25um` atlas is already cached under `~/.brainglobe/allen_mouse_25um_v1.2`, and `BrainGlobeAtlas("allen_mouse_25um")` downloads it if missing). Inputs: `data/precomputed/units_all.parquet` from [precompute-aligned-spikes](../precompute-aligned-spikes/README.md) and `data/precomputed/context_baseline_shifts.parquet` from [context-baseline-shift-analysis](../context-baseline-shift-analysis/README.md) (for the demo colouring).

Verified facts: `atlas.mesh_from_structure("root")` returns a `meshio` Mesh with `points` (49,324 x 3, float, micrometres) and one `triangle` cell block (98,638 x 3). Atlas orientation is `asr`: axis 0 = anterior-to-posterior (root mesh spans -16.8 to 13,192.5 um), axis 1 = superior-to-inferior (133.9 to 7,564.2 um), axis 2 = right-to-left (485.7 to 10,890.6 um; midline about 5,700). `atlas.annotation` (the 25 um label volume) loads in about 0.1 s and `atlas.structure_from_coords((ap, dv, ml), microns=True, as_acronym=True)` returns the structure acronym at a point; a containment check with it agrees with the units' `structure` labels for 97 to 100% of units per structure. The 61 region meshes total about 203k vertices / 404k faces (about 7 MB) plus about 1.8 MB for root. Unit coordinates `ccf_ap` (1,950 to 9,150), `ccf_dv` (500 to 5,350), `ccf_ml` (1,075 to 6,100) are in the same frame (ap -> axis 0, dv -> axis 1, ml -> axis 2), so no transform beyond axis assignment is needed. Region meshes exist for all 61 unit structures except the label `out of brain`; `atlas.structures[acr]` gives `id`, `name`, `rgb_triplet`; `atlas.get_structure_ancestors(acr)` gives the hierarchy. `trimesh`/`pyvista` are not installed.

1. **Write `code/fetch_ccf_mesh.py`** (argparse: `--atlas allen_mouse_25um`, `--out-dir` default `config.PRECOMPUTED_DIR / "ccf_meshes"`, `--structures` default = all structures present in `units_all.parquet` except `out of brain`, `--decimate-target 0.25`). Load the atlas with `BrainGlobeAtlas(name, check_latest=False)`. For `root` and each requested structure save `<out-dir>/<acronym>.npz` with `vertices` (float32, um, atlas axis order ap/dv/ml) and `faces` (int32, triangles) and write `<out-dir>/index.json` listing per structure: `acronym`, `id`, `name`, `rgb`, `n_vertices`, `n_faces`, `file`, plus `atlas`, `orientation`, `resolution_um`, `axis_order: ["ap","dv","ml"]`. Skip (and record in the index under `missing`) structures whose mesh is unavailable. Decimation: as a one-off executor action (not inside the script) run `/opt/miniconda3/bin/python3 -m pip install fast-simplification`; the script only tries `import fast_simplification`, and if it works, also save `root_decimated.npz` reduced to `decimate_target` of the faces via `fast_simplification.simplify(vertices, faces, target_reduction=1 - decimate_target)` and record it in the index; if the install or import fails, record `decimated: false` and continue (the full 98k-face root mesh is acceptable for plotly). Add `brainglobe-atlasapi>=2.3` (and `fast-simplification==<installed version>` if used) to `requirements.txt`.

2. **Write `code/brain3d.py`** with:
   - `load_mesh(acronym, mesh_dir=config.PRECOMPUTED_DIR/"ccf_meshes", prefer_decimated=True) -> (vertices, faces)`; `load_mesh_index(mesh_dir) -> dict`.
   - `to_plot_coords(ap, dv, ml) -> (x, y, z)` implementing the documented convention: `x = ml` (right hemisphere at low x), `y = -ap` (anterior towards the viewer / positive y is anterior after negation, so posterior is at the back), `z = -dv` (dorsal up). Document this in the module docstring and apply the same function to mesh vertices and unit dots.
   - `brain_figure(units: pd.DataFrame, color_col: str | None = None, hover_cols=("unit_id","structure","session_id"), selected_unit: tuple[str,int] | None = None, region_meshes: list[str] = (), mesh_opacity=0.08, region_opacity=0.15, colorscale="RdBu", color_range=None, marker_size=3, title=None) -> plotly.graph_objects.Figure`: adds `go.Mesh3d` for root (light grey, `flatshading=True`, `hoverinfo="skip"`), one `Mesh3d` per requested region (atlas `rgb`, `region_opacity`, `name=acronym`), a `go.Scatter3d` of units (drop rows with NaN coordinates and report the count in a figure annotation), coloured by `color_col` with a colorbar (or a single colour if None), `customdata = [session_id, unit_index]` per point and `hovertemplate` built from `hover_cols`; if `selected_unit` is given, add a second `Scatter3d` with one larger black-outlined marker for that unit. Layout: `scene.aspectmode="data"`, hidden axis ticks with axis titles `ML`, `AP`, `DV`, `uirevision="brain"` so the camera persists across updates, `margin=dict(l=0,r=0,t=30,b=0)`, `legend` at top-left.
   - `unit_click_to_key(click_data) -> (session_id, unit_index) | None` that parses a plotly click event's `customdata` (for the Shiny app).

3. **Alignment verification** in `code/check_brain3d.py` (PASS/FAIL, non-zero exit on failure). Load the atlas and `units_all.parquet`. For each of the structures `MOs`, `CP`, `AUDp`, `SSp`, `ACAd`, `ORBvl`, `VISp`: select units whose `structure` equals the acronym (for `SSp` also those starting with `SSp`), then (a) compute the fraction inside the region mesh's axis-aligned bounding box expanded by 100 um and assert >= 0.95; (b) **containment via the annotation volume**: for each unit call `atlas.structure_from_coords((ccf_ap, ccf_dv, ccf_ml), microns=True, as_acronym=True)` and count it inside if the returned acronym equals the structure, is in `atlas.get_structure_descendants(structure)`, or the structure is in `atlas.get_structure_ancestors(returned)`; assert the fraction >= 0.95 (measured 0.97 to 1.00 for all seven); (c) report, without asserting, the median nearest-mesh-vertex distance via `scipy.spatial.cKDTree` (interior units of large structures are legitimately far from the surface). Also assert that `root.npz` and all 61 structure meshes were written, and that the root mesh bounds loaded from the npz equal the bounds of `atlas.mesh_from_structure("root").points` within 1 um (and print them alongside the values in the facts above). Write the stdout to `work/ccf-brain-3d-component/data/checks.txt`.

4. **Demo**: `code/brain3d_demo.py --out work/ccf-brain-3d-component/data/demo_context_shift.html`: load `context_baseline_shifts.parquet`, plot `is_robust` units coloured by `log2_ratio` (range -3 to 3), regions `MOs`, `CP`, `AUDp` shown, with the most robust unit (smallest `p_perm`, then largest `|log2_ratio|`) as `selected_unit`; write the HTML with `fig.write_html(include_plotlyjs="cdn")` and print file size. If `kaleido` is importable, also write `demo_context_shift.png` for visual inspection; otherwise say so in Results.

5. **Run everything**: `python code/fetch_ccf_mesh.py`, `python code/check_brain3d.py`, `python code/brain3d_demo.py --out ...`. Record mesh counts, total size of `data/precomputed/ccf_meshes/`, and runtimes.

6. **Record artifacts.** Copy `index.json` and the check stdout (`checks.txt`) into `work/ccf-brain-3d-component/data/`; keep the demo HTML (and PNG if produced) there. Update `code/README.md` with the mesh fetch script, the `brain3d` API, and the coordinate convention.

Expected artifacts: `code/fetch_ccf_mesh.py`, `code/brain3d.py`, `code/check_brain3d.py`, `code/brain3d_demo.py`, `data/precomputed/ccf_meshes/{root.npz, <acronym>.npz ..., index.json}` (plus `root_decimated.npz` if decimation works), `work/ccf-brain-3d-component/data/{index.json,checks.txt,demo_context_shift.html[,demo_context_shift.png]}`, updated `code/README.md` and `requirements.txt`.

# Results

## Summary
The 3D brain component is in place. `code/fetch_ccf_mesh.py` exported 62 meshes (root plus all 61 unit structures, none missing) from the cached brainglobe `allen_mouse_25um` atlas into `data/precomputed/ccf_meshes/` (4.9 MB total, 1.3 s), including a decimated root mesh (12,334 vertices / 24,659 faces, 25% of the original 98,638 faces) produced with fast-simplification 0.2.0. `code/brain3d.py` provides `load_mesh`, `load_mesh_index`, `to_plot_coords`, `brain_figure` and `unit_click_to_key`. `code/check_brain3d.py` passed 16/16 checks: all meshes written, root bounds match the atlas within 1 µm, and for MOs, CP, AUDp, SSp, ACAd, ORBvl and VISp 100% of units fall inside the region bounding box (+100 µm) and 97.2 to 100% are inside the region in the annotation volume, confirming that `ccf_ap/dv/ml` map directly onto atlas axes 0/1/2 in micrometres. The demo wrote a standalone HTML of the 1,642 robust context-shift units coloured by log2(aud/vis) with MOs, CP and AUDp region meshes and the top unit highlighted. kaleido is not installed, so the demo script wrote no PNG; a screenshot was instead produced with headless Google Chrome using software WebGL (`--headless=new --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --screenshot`; the first attempt with `--disable-gpu` rendered only a 'WebGL is not supported' message) and inspected: the translucent brain, the three region meshes, unit dots along the probe tracks coloured by log2 ratio, and the highlighted selected unit all render as intended.

## Artifacts
- `code/fetch_ccf_mesh.py` — mesh export script (`--atlas`, `--out-dir`, `--structures`, `--decimate-target`).
- `code/brain3d.py` — reusable plotly figure module (API documented in `code/README.md`).
- `code/check_brain3d.py` — export and alignment checks; exits non-zero on failure.
- `code/brain3d_demo.py` — standalone demo writer.
- `data/precomputed/ccf_meshes/root.npz`, `root_decimated.npz`, 61 `<acronym>.npz`, `index.json`.
- `data/index.json`, `data/checks.txt`, `data/demo_context_shift.html` (under this work folder).
- `data/demo_context_shift.png` (under this work folder) — headless Chrome screenshot of the demo for visual inspection.
- `code/README.md` updated (fetch script, brain3d API, coordinate convention); `requirements.txt` gained `brainglobe-atlasapi>=2.3` and `fast-simplification==0.2.0`.

## Alignment check numbers

| structure | n units | in bbox (+100 µm) | in annotation volume | median nearest surface vertex |
|---|---|---|---|---|
| MOs | 1150 | 1.000 | 0.999 | 300 µm |
| CP | 462 | 1.000 | 1.000 | 243 µm |
| AUDp | 312 | 1.000 | 1.000 | 184 µm |
| SSp* | 639 | 1.000 | 0.972 | 326 µm |
| ACAd | 157 | 1.000 | 0.994 | 97 µm |
| ORBvl | 339 | 1.000 | 1.000 | 121 µm |
| VISp | 57 | 1.000 | 1.000 | 89 µm |

## Step-by-step
1. `fetch_ccf_mesh.py` written and run: 62 meshes, 0 missing, decimation succeeded (one-off `pip install fast-simplification` done beforehand as the plan specified); versions pinned in `requirements.txt`.
2. `brain3d.py` written with the documented convention `x = ml`, `y = -ap`, `z = -dv`; root mesh light grey at opacity 0.08, regions at 0.15 with atlas colours, units as `Scatter3d` with `customdata = [session_id, unit_index, hover cols]`, selected unit as a larger yellow marker with black outline, `uirevision="brain"`, NaN-coordinate units dropped and counted in an annotation.
3. `check_brain3d.py` written and run: 16/16 PASS (table above); the reviewer's notes were followed (NaN-coordinate unit dropped first; ancestor lookup guarded against labels that are not valid acronyms).
4. `brain3d_demo.py` written and run: HTML 2.07 MB with plotly from CDN. PNG via headless Chrome with SwiftShader WebGL (see Summary).
5. Runtimes: fetch 1.3 s, checks about 10 s (dominated by per-unit annotation lookups), demo about 2 s.
6. Artifacts copied; `code/README.md` updated.

Deviations: none. The one NaN-coordinate unit in `units_all.parquet` is excluded from the checks and from plotting, as the reviewer suggested.

# Assessment

## Verdict
accomplished

## Reasoning
The Goal asked for a reproducible fetch of the Allen CCF root mesh (plus optional region meshes) into `data/precomputed`, a reusable plotly figure function with a translucent brain, colour-coded unit dots with hover text and a selected-unit highlight, a verified and documented coordinate convention, and a standalone HTML demo. All of these exist and were independently re-verified read-only: `data/precomputed/ccf_meshes/` holds 63 npz files plus `index.json` (root, `root_decimated`, and all 61 unit structures, `missing: []`); `root.npz` is 49,324 x 3 float32 vertices / 98,638 x 3 int32 faces with bounds (-16.8, 133.9, 485.7) to (13,192.5, 7,564.2, 10,890.6) um, and `root_decimated.npz` is 12,334 / 24,659 with matching bounds, exactly as stated in Results. Importing `brain3d` and calling `brain_figure` on a 200-row subset of `units_all.parquet` with `region_meshes=["MOs"]` and a `selected_unit` returned a `plotly.graph_objects.Figure` with traces `mesh3d root`, `mesh3d MOs`, `scatter3d units (n=200)`, `scatter3d selected unit`, `customdata` beginning `[session_id, unit_index, ...]`, `scene.aspectmode="data"` and `uirevision="brain"`; with a `color_col` the colorbar is enabled. `unit_click_to_key({"points":[{"customdata":["759434_2025-02-04", 5, ...]}]})` returned `("759434_2025-02-04", 5)` and a malformed event returned `None`. Re-running `code/check_brain3d.py` reproduced 16/16 PASS with the same numbers as `data/checks.txt` (bbox fraction 1.000 and annotation containment 0.972 to 1.000 for the seven structures), exit code 0. `requirements.txt` contains `brainglobe-atlasapi>=2.3` and `fast-simplification==0.2.0`, both matching the installed versions.

Plan adherence is complete: every numbered step and every expected artifact is present, `index.json` in the work folder is byte-identical to the one in `data/precomputed/ccf_meshes/`, the demo HTML (2.07 MB, plotly from CDN) exists, and `code/README.md` documents the fetch script, the `brain3d` API and the `x = ml, y = -ap, z = -dv` convention. The kaleido fallback was handled as the plan allowed (no PNG from the script) and the executor went beyond it by producing a headless-Chrome screenshot; that PNG shows the translucent brain, the MOs/CP/AUDp region meshes, 1,642 unit dots along probe tracks coloured by log2(aud/vis) with the RdBu colorbar, and the highlighted yellow selected unit, so the visual claim in Results is supported. The only discrepancy noticed is trivial: the check script ran in about 1 s here versus the "about 10 s" reported, most likely atlas caching.

## Recommended next status
done

# Review Notes
Approved on attempt 2: independently re-verified the root mesh bounds (-16.8 to 13,192.5 um on axis 0), the annotation-volume containment check (0.972 to 1.000 for the seven structures, bbox fraction 1.0), that all 61 non-`out of brain` structures have meshes, and that `get_structure_descendants`/`get_structure_ancestors` and the input parquet columns exist; the fast-simplification and kaleido fallbacks are correctly stated as optional. Minor executor notes: drop the one unit with NaN coordinates before the step 3 checks, and guard the ancestor lookup in check (b) against `structure_from_coords` returning `'Outside atlas'` (not observed for these structures, but it is not a valid acronym).
