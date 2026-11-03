# Integration Plan: Striatal Feedback + Behavior ↔ Auditory Representational Dimensionality + Spectrotemporal Tuning

Companion to `REVISION_PLAN.md`. Scope: connect this repo's striatal-feedback / behavioral-model
findings with the auditory representational-geometry work in the sibling repos (`FLT2`, `FST1`), as
one coherent mechanistic story.

**Phased, per decisions with the user (2026-09-10):**
- **Phase 1 — goes in the revision:** analysis **B** only (striatal feedback trend ↔ auditory-cortex
  representational dimensionality). One pre-registered link, one figure panel, one paragraph.
- **Phase 2 — same manuscript, later, on the user's go:** the SPC-strategy links, the FST
  spectrotemporal-baseline analyses, whole-brain / language-group extensions.
- Representational dimensionality is **recomputed on fMRIPrep 25.2.5, in native T1w space** to match
  this repo's (`FLT1`'s) revised pipeline — not caveated. See Workstream I.
- `FST1`/STgrid and `FLT2`/tonecat are the **same participants, same MRI sessions**; the 25.2.5
  fMRIPrep run covers **both tasks**.
- Integration sample = the **12 non-Mandarin** subjects analyzed in the manuscript
  (`FLT02, 04, 06, 09, 11, 12, 13, 14, 20, 25, 28, 30`); n=11 for behavior (`FLT02` dropped —
  corrupted response coding in an early run).

**No code changes yet — this is a roadmap.**

## Why do this

Both reviewers asked for integration, not more standalone analyses. The manuscript's throughline is
*striatal feedback processing during auditory category learning* + *a behavioral shift toward
procedural (SPC) strategy*. The sibling repos characterize *auditory cortical/subcortical
representational geometry* in the same participants. Linking them tests one mechanistic claim: **as
learning proceeds, the striatal feedback signal (aCAU decline) tracks a reorganization of auditory
representations toward a lower-dimensional, more categorical code.** Phase 1 establishes the
striatum↔cortex leg of that claim — the linkage R2 explicitly proposed, drawn between the two
results R2 already accepted as the paper's core. Phase 2 closes the triangle to behavior and adds
the FST acoustic baseline.

## Inventory — what each repo provides

Same FLT cohort, same subjects, same sessions across all three (loop scripts iterate the same
`data_denoised/sub-*` list).

| Source | Analysis | Key per-subject outputs | Stage split? | Space / betas |
|---|---|---|---|---|
| **`fMRI_auditory-category-learning`** (this repo) | Striatal univariate GLM | `fb_correct − fb_wrong` and sound betas per Tian-S2 ROI (aCAU/pCAU/aPUT/pPUT/NAc) + auditory + PFC; per-subject linear learning-stage trend scalar (weights `[-1,0,1]`) | Yes (early/middle/final) | T1w, nilearn condition-level GLM (`scrubbed`+collapsed-nuisance), fMRIPrep 25.2.5 |
| ″ `08_behavioral-modeling/` | SPC/decision-bound model comparison | per-subject `late_minus_early` Δ-BIC (SPC vs. best competitor) | early vs. final | n/a (behavioral) |
| ″ `07_rsa/` | RSA (striatum + cortex), rsatoolbox | per-ROI RDMs, Tone/Talker model fits | run-grouped | nilearn per-run LSS betas, fMRIPrep 22.1.1 — **cut from the manuscript once already** |
| **`FLT2`** | Representational dimensionality | `sub-*_dimensionality_PR.csv` (participation ratio), `_Dsheng.csv`, `_eigenspectra.npz`, per stage in `stage-{early,middle,final}/` | **Yes (same early/middle/final)** | GLMsingle trial betas, 16 category stimuli (4 tones × 4 talkers); auditory-pathway ROIs (IC, MGN, HG, PT, PP, STGp, STGa, ParsOp, ParsTri, bilateral); **currently MNI — to be recomputed in T1w** |
| ″ `representational_dimensionality.ipynb` | Group-level | §8e stage effect on PR/D_sheng; §9i relates dimensionality to behavioral accuracy; §9 Mandarin vs non-Mandarin; §11c learning-stage searchlight | — | — |
| **`FST1`** | STgrid spectrotemporal task | voxelwise spectral × temporal modulation tuning maps (`analyze_stgrid_spectrotemporal.py`, R²-thresholded), PR + eigenspectra for STgrid stimuli, RSA | **No — single session** | **currently MNI — to be recomputed in T1w**, GLMsingle; same auditory-pathway ROIs |

## What's directly comparable

- **Subjects**: the same 12 (confirmed — FST and FLT are the same sessions).
- **ROIs**: auditory-pathway ROIs (HG/PT/PP/STG/Pars, + IC/MGN) exist in `FLT2`/`FST1`; striatal
  ROIs (aCAU/aPUT/…) in this repo. Auditory-cortex ROIs also exist in this repo's
  `group_level_all_ROI.ipynb` `region_dict['auditory']` — the overlap set for any cortex↔cortex work.
- **Learning-stage structure**: identical early/middle/final run grouping in this repo's univariate
  GLM **and** `FLT2`'s dimensionality — per-stage and trend-scalar comparisons line up 1:1.
- **Per-subject scalars to correlate** (one number per subject):
  1. aCAU feedback learning-stage trend (this repo)
  2. aPUT feedback trend (this repo)
  3. aCAU − aPUT trend difference (this repo — the headline scalar)
  4. SPC relative-BIC `late_minus_early` (behavioral)
  5. auditory-ROI PR (or D_sheng) learning-stage trend (`FLT2` — the same `[-1,0,1]` contrast on the
     three `stage-*` values)
  6. auditory-ROI Tone/Talker RSA fit trend (`07_rsa`, Phase 2 only, and only if deliberately re-added)
  7. `FST1` spectrotemporal tuning sharpness / STgrid PR — a **stable trait**, not a trend

---

## Phase 1 — analysis B (the link for the revision)

### B, defined

**Striatal side** (this repo, already on 25.2.5): per-subject `[-1,0,1]` learning-stage trend of the
`fb_correct − fb_wrong` contrast, from `roi_values_long.csv` / `robustness_summary.csv`. Two scalars:
- aCAU feedback trend
- aCAU − aPUT differential trend (the double-dissociation scalar)

**Cortical side** (`FLT2`, recomputed on 25.2.5 in T1w — Workstream I): per-subject `[-1,0,1]`
learning-stage trend of auditory-cortex **PR**, computed on the three `stage-*` PR values (mirrors
the striatal scalar exactly). Pre-registered ROI choices:
- *main*: Primary auditory composite (HG + PT + PP, L/R pooled) PR trend — motivated by the
  representational-compression hypothesis (tone categories sharpen earliest in primary cortex)
- *secondary a-priori*: R-PP alone (`FLT2` §8e-ii flagged it as the strongest nominal early-vs-final
  effect) — one pre-specified test, no correction
- *specificity check*: Association composite (STGa/STGp/ParsOp/ParsTri) PR trend — expected null

Fix the **expected sign** of the cortical PR trend from `FLT2`'s existing group-level stage effect
(§8e) *before* running B, so the correlation sign isn't chosen post hoc.

### Test

Per-subject correlation (Pearson; Spearman as robustness) between each striatal scalar and the
cortical PR trend, n=12, with the `resampling_stats.py` battery already at repo root — bootstrap 95%
CI (10k), LOSO (12 folds), exact sign-flip permutation. Stats table via `stats_fmt.py`.

**Prediction**: subjects with the steepest aCAU feedback learning-stage decline show the strongest
auditory-cortex PR change over learning (representational compression as categories consolidate).

### Output

One scatter panel (12 points, LOSO-stable fit + CI band) + one paragraph in Results/Discussion tying
the two core results: striatal feedback teaching signal ↔ cortical representational refinement.

---

## Workstream I — recompute `FLT2` + `FST1` dimensionality on fMRIPrep 25.2.5, in T1w space

**Status (2026-09-15): steps 2–4 done in code, and the cluster run is now underway with most of
it working.** `FLT2` tonecat GLMsingle → masking → `dimensionality_roi.py` has completed
successfully for the large majority of the cohort (23 of ~25 subjects in the notebook's current
`sub_list`, run broader than just the 12 non-Mandarin — harmless superset for Phase 1, which only
needs the 12). `FST1` stgrid GLMsingle has also been run; masking/dimensionality for it are next.
Getting `FLT2` running surfaced and fixed several real bugs beyond the original code-readiness
work (see step 5) — this is normal "first real run" fallout, not scope creep, but it means the
step 2–4 "done in code" status from 2026-09-14 undersold what was still broken. Two subjects
(FLT02, FLT22) are being rerun now with all fixes applied; FLT07 remains the known/expected bad-QA
exclusion. Step 4's reference-image caveat was investigated and the shared masking script
deliberately left task-agnostic (see step 4) — the block-design finding in step 5 makes this less
urgent than it looked (see below), but the affine/shape check is still open.

Per repo (`FLT2`/tonecat first — Phase 1 only needs this; `FST1`/stgrid can lag to Phase 2). The
goal: `FLT2`/`FST1` dimensionality is derived from the **same fMRIPrep version, same native T1w
space, and the same per-subject ROI masks** as this repo's striatal univariate stats. The only
remaining methodological difference from `FLT1` is the trial-beta estimator (GLMsingle vs nilearn
LSS), which is deliberate and stated in Methods.

1. 🔲 **Checklist.** Confirm the 25.2.5 derivative tree has, for all 12 subjects and **both tasks**:
   confounds TSV, **`space-T1w` preproc BOLD** (i.e. `T1w` was in the 25.2.5 `--output-spaces`), and
   the anat `from-MNI152NLin2009cAsym_to-T1w` transform. Confirm the T1w `carpet_dseg` masks from
   this repo's Workstream 8 (`05_masking/warp_atlases_to_T1w.sh` +
   `make_atlas_region_masks.py --space=T1w --atlas=carpet_dseg`) exist for all 12 and cover the
   auditory ROI set (HG/PT/PP/STGa/STGp/ParsOp/ParsTri). **Still open: confirm the tonecat/stgrid
   shared-grid assumption directly** (e.g. `nibabel` affine + shape for one `task-tonecat` and one
   `task-stgrid` `space-T1w` BOLD file per subject) — this is what step 4's mask reference-image
   caveat actually depends on, and it's a five-minute check rather than a guess (somewhat
   de-risked now that both tasks' GLMsingle runs have completed without space/grid errors, but not
   the same as actually checking). **RESOLVED (2026-09-15): the FLT06/FLT11/FLT14 gap flagged
   below is fixed.** All three now have real `FLT2` dimensionality output from the actual cluster
   run — the updated loop scripts (12-subject list) did bring them back in, and their GLMsingle fit
   completed without erroring. Original flag, kept for the record: `FLT2`'s previous
   `loop_mask_glmsingle_betas.sh` hard-coded an 18-subject list that excluded
   FLT06/FLT08/FLT11/FLT14/FLT18/FLT26 from this repo's GLMsingle masking pipeline entirely, for a
   reason never documented in either repo — three of those are in the manuscript's 12-subject
   non-Mandarin sample. Still worth a QA glance at these three specifically before fully trusting
   their PR values (a clean run isn't the same as confirmed-good data), but there's no longer a
   missing-data blocker.
2. ✅ **Env + paths — done.** `/bgfs/` → `/ix1/`; `denoised_fmriprep-22.1.1` →
   `denoised_fmriprep-25.2.5`; `--space=MNI152NLin2009cAsym` → `--space=T1w`; `run_*.sh`
   `bidsroot`/`fmriprep_dir` updated in both repos, both tasks. *(Still needed: pin the conda env
   to the manuscript `environment.yml` on the cluster — not a code change.)*
3. ✅ **Version + space tag the GLMsingle output dir — done.** Added a `--glmsingle_dirname` CLI
   arg (default `glmsingle` / `glmsingle_stgrid`, unchanged for old callers) to
   `modeling_firstlevel_GLMsingle.py`/`_stgrid.py`, `mask_glmsingle_betas.py`, and
   `dimensionality_roi.py` in both repos; `run_*.sh` now pass
   `glmsingle_fmriprep-25.2.5_space-T1w` / `glmsingle_stgrid_fmriprep-25.2.5_space-T1w` so the
   recompute can't collide with the untagged 22.1.1/MNI outputs. `loop_mask_glmsingle_betas.sh`
   (both repos) now targets the 12 non-Mandarin subjects directly (see step 1's flag);
   `loop_dimensionality_roi.sh`/`loop_modeling_firstlevel_GLMsingle_stgrid.sh` just needed the
   `/ix1/` path (they already auto-discover subjects from `bidsroot`).
4. ✅ **Reuse this repo's T1w ROI masks — confirmed, no code change needed.** Traced
   `mask_glmsingle_betas.py`'s existing mask-path construction
   (`{mask_dir}/sub-{sub}/space-{space}/masks-{atlas}/sub-{sub}_space-{space}_mask-{roi}.nii.gz`)
   against `05_masking/make_atlas_region_masks.py`'s T1w `carpet_dseg` branch — **same path
   template, same filename pattern, same `roi_dict_MNI_dseg` label dictionary, same source atlas
   file.** Passing `--space=T1w --atlas=dseg --mask_dir=$bidsroot/derivatives/nilearn/masks/`
   (already in `run_mask_glmsingle_betas.sh`) reads this repo's Workstream-8 masks directly. (IC/MGN
   would come from the `subcort_aud` T1w atlas, also already warped, but `FLT2` excludes them for
   tSNR/voxel-count reasons — no action.) **Caveat, considered and deliberately left alone:**
   `make_atlas_region_masks.py` resamples onto a single functional reference image per subject
   (whichever task's file sorts first) and there is only **one** mask directory per subject, shared
   by this repo's own tonecat-based analyses *and* `FLT2`/`FST1`'s GLMsingle masking — not a
   separate mask tree per task. An earlier pass at this plan tried anchoring that reference to a
   `task-stgrid` file specifically; reverted (2026-09-14, prompted by the user catching it) because
   it only relocates the risk — if tonecat and stgrid don't actually share an acquisition matrix,
   biasing the *shared* default toward stgrid would put the mismatch onto this repo's own manuscript
   results instead of removing it, for a script whose primary consumer has nothing to do with
   stgrid. The right fix if the grids ever turn out to differ is separate per-task mask trees, not a
   different default reference file — so for now: confirm tonecat and stgrid BOLD share an
   acquisition matrix per subject (e.g. compare `nibabel` affine/shape for one tonecat and one
   stgrid `space-T1w` file per subject) before fully trusting `FST1` ROI betas against these masks
   (flagged
   inline in `FST1/run_mask_glmsingle_betas.sh`).
5. 🟡 **Re-run GLMsingle first-level — run, mostly successful, several real bugs found and fixed
   in the process (2026-09-14 → 2026-09-15):**
   - **`FLT2` wasn't saving betas as NIfTI at all** — `GLM_single.fit()`'s own `.npy` outputs
     aren't what `mask_glmsingle_betas.py`/`dimensionality_roi.py` read. Ported `save_betas_as_nifti`/
     `save_r2_as_nifti` from `FST1` into `FLT2/modeling_firstlevel_GLMsingle.py`.
   - **`FLT2`'s design matrix collapsed repeated within-run trials into one column per condition**
     (`convert_events_by_condition`) instead of one column per individual trial — this both
     contradicts GLMsingle's own documentation/the Prince-Kay et al. 2022 *eLife* paper (its
     fracridge trial-wise regularization has nothing to shrink over otherwise) and doesn't match
     `dimensionality_roi.py`'s `rep-(\d+)` filename requirement, so masked CSVs were silently
     dropped downstream. Rewrote as `convert_events_by_trial`: one column per trial, `rep-N`
     stamped into the saved filename. Unit-tested on synthetic events before deploying.
   - **`FST1` did *not* need the same fix.** Checked an actual events.tsv: each of the 16 stgrid
     stimuli occurs *exactly once per run* as a 20s block — condition and trial coincide 1:1, so
     `FST1`'s existing condition-per-run design was already single-trial-equivalent. No rerun
     needed there.
   - **Is GLMsingle even valid for a 20s block design?** Checked directly against GLMsingle's own
     docs/wiki (not assumed): yes — block designs are an officially supported, documented use case
     (dedicated example notebook), duration is handled by the separate `stimdur` parameter (not by
     marking every TR of the block in the design matrix), and `FST1`'s script already reads
     `stimdur` dynamically from the events.tsv `duration` column rather than reusing `FLT2`'s
     hardcoded brief-trial value. Block spacing here (~20s) is well inside GLMsingle's documented
     safe range (risk starts under ~4s). No code change needed.
   - **Stale-cache bug**: `if not os.path.exists(outputdir_glmsingle)` doesn't mean "already
     completed" — GLMsingle creates that directory before it finishes, so a job interrupted
     mid-fit (in this case, by leftover non-`acq-dwidenoise` duplicate BOLD files causing
     `first_level_from_bids` to raise "too many images found" earlier — since fixed by cleaning
     the duplicates) leaves a directory with no `.npy` outputs, and every later resubmission
     crashed trying to load files that were never written. Fixed the check to look for
     `TYPED_FITHRF_GLMDENOISE_RR.npy` specifically. This is what broke FLT02/FLT22, which are
     being rerun now with the fix.
   - **`slice_time_ref`**: removed the manual `0.5*t_acq/t_r` computation from both `FLT2` and
     `FST1`, matching this repo's own `06_univariate/univariate_glm.py` (already validated:
     nilearn's BIDS-sidecar auto-inference matches the old manual formula to ~3%). Not rerun
     retroactively for the 23 already-completed `FLT2` subjects — not worth the recompute cost for
     a ~3% effect; FLT02/FLT22 pick it up automatically since they're being rerun anyway, `FST1`
     had it from the start since nothing there had run yet.
   - `.gitignore` added to `FLT2` and `FST1` (matching this repo's); 18 previously-tracked
     `.ipynb_checkpoints/`/`__pycache__` files untracked from `FLT2` (kept on disk, `git rm --cached`).
6. 🟡 **Re-mask betas** (`mask_glmsingle_betas.py`, `--space=T1w`). Done for `FLT2`'s 23 completed
   subjects; pending FLT02/FLT22 (blocked on step 5's rerun) and all of `FST1` (GLMsingle only just
   finished there). Also fixed while resolving a merge conflict against an unrelated `/bgfs/`→`/ix1/`
   migration commit already on `FST1`'s `main`: `run_mask_glmsingle_betas.sh`'s atlas loop was dead
   code (`#for atlas in tian-S3 dseg subcort-aud`, hardcoded to `dseg` only) — now actually loops
   `dseg` + `subcort-aud` (both warped to T1w in Workstream 8), `tian-S3` deliberately excluded
   since it was never built in T1w.
7. 🟡 **Re-run `dimensionality_roi.py`**. Done for `FLT2`'s 23 completed subjects →
   `sub-*_dimensionality_PR.csv` + `stage-*/`; pending FLT02/FLT22 and all of `FST1`.
8. 🟡 **Refresh `representational_dimensionality.ipynb`** group aggregation (§6–8). User has
   updated the single-subject-inspection setup cell's paths (`glmsingle_dir`, `space_label='T1w'`)
   and `sub_list`. One bug found and fixed along the way: the single-subject cell unconditionally
   loaded a searchlight D_sheng NIfTI that doesn't exist yet (`dimensionality_searchlight.py`
   hasn't been run for this recompute — Phase 2) — changed to read the ROI-level D_sheng CSV
   `dimensionality_roi.py` already produces, same as PR. Group-level cells (§6+) not yet reached —
   blocked until FLT02/FLT22 land, though nothing stops working through them with the 23 now.

Notes:
- Everything is now native T1w — B correlates a cortical-T1w PR trend with a striatal-T1w feedback
  trend, same fMRIPrep version, same mask provenance. No cross-space caveat needed.
- **Whole-brain / surface figures still need MNI** (Phase 2): `FLT2`'s `dimensionality_searchlight.py`
  runs per-subject (fine in T1w), but group aggregation of PR/D_sheng maps and any
  `vol_to_surf`/fsaverage rendering need a shared grid — the same constraint this repo hit in
  Workstream 8 (#17/#20). Phase 2 would warp per-subject T1w searchlight maps to MNI for the group
  step, or run the searchlight in MNI purely for that figure.
- Cluster cost: GLMsingle ≈ 8 h / subject, 48 G, ×12 (×2 tasks for full Phase 2).
- The 22.1.1/MNI outputs stay in place; the recompute writes to the parallel version+space-tagged tree.

## Workstream II — merge layer (`09_integration/`)

1. `subjects.csv` — the 12, with per-repo "dimensionality computed on 25.2.5?" status and language
   group.
2. `roi_crosswalk.csv` — Tian S2 ↔ auditory `dseg` ↔ carpet label mapping.
3. Exporters:
   - this repo: per-subject aCAU / aPUT / aCAU−aPUT feedback trend → a small CSV
     (`derivatives/nilearn/group_level/roi_trend_scalars.csv`), from a cell added to
     `group_level_all_ROI.ipynb` or derived in the integration notebook from `roi_values_long.csv`.
   - `FLT2`: the integration notebook reads `sub-*_dimensionality_PR.csv` directly (cross-repo read,
     like `FLT2` §9i does in reverse).
4. `09_integration/integration_B.ipynb` — merge on `participant_id`, compute the cortical PR trend
   summaries, run the correlations + `resampling_stats.py` battery, emit the figure panel + stats
   table.

## Phase 2 hooks (documented, not built)

All reuse Workstream I (dimensionality recomputed on 25.2.5 in T1w, incl. FST/stgrid) + the
Workstream II merge layer, so building Phase 1 de-risks them.

- **A. SPC strategy ↔ representational geometry.** aCAU feedback trend ↔ SPC `late_minus_early`
  Δ-BIC (this is `REVISION_PLAN.md` Workstream 1's neural-integration analysis — already planned
  separately); and SPC `late_minus_early` ↔ auditory PR trend (behavior ↔ cortex, independent of
  striatum — closes the triangle).
- **C. FST spectrotemporal baseline.** (i) `FST1` spectrotemporal tuning sharpness / STgrid PR as a
  **trait moderator/covariate** in the Phase 1 and A links — do subjects with a sharper or more
  efficient baseline auditory code show faster representational compression and/or stronger striatal
  learning signals? (ii) within-subject paired contrast: PR per auditory ROI, FST (acoustic grid)
  vs FLT (learned categories) — is category learning associated with lower effective dimensionality
  in the same ROI/subject?
- **RSA (`07_rsa`), only if load-bearing.** Does the Tone-model RDM fit increase across stages, and
  does that increase track aCAU feedback decline? RSA was cut once for "drift" — only re-add if it's
  central to the story, not as a bonus. Would need its own nilearn-LSS recompute on 25.2.5.
- **Mandarin vs non-Mandarin.** `FLT2` §9 already does this on dimensionality. A matched
  Mandarin/non-Mandarin split on the striatal feedback trend would give a parallel contrast — but
  that pulls Mandarin subjects back into a pipeline scoped to exclude them; weigh whether it's worth
  it.

## Reconciliation costs / risks

- **Beta pipeline mismatch.** `FLT2`/`FST1` use **GLMsingle** trial-level betas; this repo's
  striatal stats use **nilearn** condition-level GLM. Not fatal for a subject-level scalar
  correlation (each side is internally consistent), but state it plainly in Methods and don't imply
  a single unified model. **Refined (2026-09-15), now that both `FLT2` and `FST1` GLMsingle runs
  are real, not just code:** `FLT2` produces true single-trial betas (`rep-N`, ~3 repeated
  presentations per stimulus per run — the fix in Workstream I step 5). `FST1` produces one beta
  per stimulus per run, which is *also* single-trial in effect, not a simplification — each of the
  16 stgrid stimuli occurs exactly once per run as a 20s block, so condition and trial coincide.
  So despite the different code paths, both sides give GLMsingle one column per true trial; the
  earlier open question about whether `FLT2`/`FST1` were methodologically inconsistent for a
  Phase 2 FST-vs-FLT comparison is resolved — they aren't.
- **Space.** Resolved by Workstream I — `FLT2`/`FST1` dimensionality recomputed in native T1w with
  this repo's own warped ROI masks, so ROI-level scalars are directly comparable. Only *voxelwise* /
  searchlight cross-references (Phase 2 whole-brain, surface figures) still need a common MNI grid.
- **Causality.** Single cohort, correlational, n=12 — frame everything as "consistent with /
  co-varies with," never "drives." The DWI gradient (independent HCP sample) motivates
  directionality a priori but can't test it here.
- **Multiple comparisons.** `REVISION_PLAN.md` Workstreams 4/5 exist to *not* multiply exploratory
  tests. Phase 1 is one pre-registered link reported with the robustness checks. Phase 2 stays a
  small pre-specified set, or it re-creates the exact problem the revision is fixing.
- **FLT06/FLT11/FLT14 — resolved (2026-09-15).** Previously flagged as having zero `FLT2` GLMsingle
  output for an undocumented exclusion reason; the actual cluster run confirms all three now have
  real dimensionality output. Kept here for the record, not as an open risk.

## Sequencing

1. ✅ **Code ready.** Version+space-tagged the GLMsingle output dir (`--glmsingle_dirname`);
   confirmed masking needs no code change (`--space=T1w --atlas=dseg` already reads this repo's
   Workstream-8 masks); path/env updates in `FLT2` + `FST1` (`/ix1/`, 25.2.5, `--space=T1w`), both
   tasks. Subject-loop scripts retargeted to the 12 non-Mandarin (flagging FLT06/11/14, above).
2. ✅ **Effectively done via the real run**, not a separate pre-check. `space-T1w` BOLD exists for
   both tasks and `warp_atlases_to_T1w.sh`'s masks are usable — both GLMsingle runs and masking
   completed without space/grid errors. Still open: the tonecat/stgrid shared-acquisition-matrix
   spot-check itself was never done directly (see step 1's note) — a clean run is reassuring but
   not the same thing.
3. 🟡 **In progress.** `FLT2` tonecat GLMsingle → mask → `dimensionality_roi.py`: done for 23 of
   ~25 subjects (run broader than just the 12 non-Mandarin — harmless superset). FLT02/FLT22 being
   rerun now with the step-5 bug fixes; FLT07 is the known bad-QA exclusion. Several real bugs
   found and fixed getting here — see Workstream I step 5.
4. 🔲 Refresh `FLT2` `representational_dimensionality.ipynb` group-level cells (§6+) on the full
   set once FLT02/FLT22 land; lock the expected PR-trend sign from §8e. (Single-subject setup cell
   already updated/fixed — see step 8.)
5. 🔲 Workstream II: `subjects.csv`, `roi_crosswalk.csv`, exporters, `integration_B.ipynb`.
6. 🔲 Run B's correlations + robustness; draft the paragraph + figure; log into `REVISION_PLAN.md`
   Workstream 1.
7. 🟡 **`FST1` stgrid GLMsingle already run**, ahead of the original "Phase 2, on the user's go"
   sequencing — no harm, doesn't block Phase 1. Masking/dimensionality for it are next (step 6/7).
   Analyses A and C, RSA / language-group extensions remain deliberately on hold.

## To confirm as we go

- Canonical striatal scalar for B: `fb_correct − fb_wrong` final-vs-early (default — matches the
  manuscript headline) vs `fb_correct` alone.
- ✅ **Resolved**: GLMsingle confound handling left as-is (no external `load_confounds` call before
  it) — confirmed still true after the step-5 fixes; the only change from the 22.1.1/MNI outputs is
  fMRIPrep version + space, the cleaner comparison for Phase 1.
- ✅ **Confirmed**: the 25.2.5 fMRIPrep run has `space-T1w` BOLD for both tonecat and stgrid, and
  `warp_atlases_to_T1w.sh`'s masks work — both GLMsingle runs completed against them successfully.
- ✅ **Confirmed**: GLMsingle behaves correctly on native-T1w BOLD, including at 20s-block-design
  temporal structure for `FST1` (checked directly against GLMsingle's own documentation — block
  designs are officially supported via `stimdur`).
- Still open: the tonecat/stgrid shared-acquisition-matrix affine/shape spot-check (step 1) —
  never actually run, just de-risked by both tasks' GLMsingle completing without grid errors.
- Still open: FLT02/FLT22 completion; a QA glance at FLT06/FLT11/FLT14 specifically (clean run,
  not yet confirmed clean *data*).
- Lock the cortical PR-trend direction from `FLT2` §8e before running B.
- That GLMsingle behaves on native-T1w BOLD as expected (it is space-agnostic — operates on the
  timeseries matrix — but confirm memory/runtime at the larger native voxel count).
- Lock the cortical PR-trend direction from `FLT2` §8e before running B.
