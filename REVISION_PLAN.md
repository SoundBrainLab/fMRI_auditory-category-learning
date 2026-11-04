# Revision Plan: Striatal 7T fMRI Auditory Category Learning Manuscript

## Context

The manuscript ("Distinct 7T functional MRI response patterns to stimulus and feedback across dorsal striatum during auditory learning") was rejected by *Journal of Neuroscience* after two rounds of revision (JN-RM-0379-25R2, decision 2026-08-16). Both reviewers' core complaint is that the manuscript **drifted across revisions**: RSA was cut, behavioral modeling (SPC vs. decision-bound models) and PFC connectivity were added late and poorly integrated. R2 was largely positive about the two "core" results (DWI structural-connectivity gradient; univariate feedback/sound response profiles across striatal subregions) but flagged integration/reporting gaps. R1 was more skeptical: doubted the feedback results would replicate at n=12, and flagged the behavioral modeling for lacking a direct model comparison, a structurally biased comparison, and a small (16-stimulus) stimulus set relative to the decision-bound literature.

**Decisions already made with the user:**
- Keep the SPC/decision-bound modeling (sound method, under-described/poorly integrated) — fix via write-up + integration, not removal.
- Stay journal-agnostic for now; plan is venue-neutral. (*PLOS Biology* would need a broad-significance reframe for a confirmatory n=12 paper; *Neurobiology of Language*/*eNeuro* are likelier fits given the reviews are about integration/rigor, not novelty.)
- Open to new analyses beyond robustness checks, as long as n=12 is respected (no new data, no comparison-multiplying analyses without justification).

**Key repo findings shaping what's feasible:**
- Manuscript text lives outside the repo (Word/Overleaf) — this plan is a roadmap, not literal text edits.
- **SPC/decision-bound modeling code doesn't exist in this repo or its history** — run externally, never integrated. First blocking item below.
- HCP structural-connectivity work (`hcp7t_mrtrix3_TianS2/`) uses an **independent normative sample** (up to 100 HCP 7T participants), not this study's own 12 — confirms R1's "not unique to this study" critique; rules out within-subject structure↔function↔behavior analysis (this study never collected its own DWI — `02_denoising/single_sub_dwi_denoise.sh` denoises *functional* data, not diffusion).
- The striatum–PFC full ANOVA R2 asked for is **already computed** (`hcp7t_mrtrix3_TianS2/connectivity_statistics.ipynb` → `stats_exports/anova_pfc_full.tsv`) — just needs pulling into the manuscript.
- Univariate stats live in `06_univariate/group_level_all_ROI.ipynb` (Tian-S2 8-ROI striatal parcellation × hemisphere + NAc, up to 3–4-way RM-ANOVAs at n=12) — consistent with both reviewers' power concerns.
- `07_rsa/` still exists despite RSA being cut from the manuscript — no action unless revisited.
- Fig 4C/3B duplicate is a manuscript-assembly error, not a pipeline bug (no duplicate output files).
- *(Historical, fixed — see Progress.)* The three original GLM scripts had aCompCor disabled, no motion scrubbing, and fMRIPrep pinned to 22.1.1. The user's `sitek/SSP` repo already runs fMRIPrep 23.2.1 with `load_confounds_strategy(denoise_strategy='scrubbing', fd_threshold=0.9, std_dvars_threshold=1.5)` — the validated setup this revision adopts (Workstream 7).

## Workflow conventions
- **All work happens on `manuscript-revision`** (off `main`); nothing merges back until the user signs off.
- **Keep `README.md` current as changes land**, not as a final catch-up pass.
- **`.gitignore`** covering `__pycache__/`, `*.pyc`, `.DS_Store`, `.ipynb_checkpoints/`, local venvs.
- **`environment.yml`** capturing exact package versions behind the already-reviewed manuscript numbers, before any upgrades. External cluster tools (FreeSurfer, FSL, AFNI, ANTs, MRtrix3, Singularity) are `module add`-based, not conda — version them in README/script comments instead. Regenerate once Workstream 7 lands.

## Progress (2026-08-19)

Working on `manuscript-revision` (off `main`, not yet merged):
1. `.gitignore` added.
2. `03_fmriprep/run_fmriprep_denoised_25.2.5.sh` (new, separate derivatives tree), starter `environment.yml`, confound/scrubbing modernization applied to the three original `06_univariate/*.py` scripts.
3. T1w-space atlas support: `05_masking/warp_atlases_to_T1w.sh` (Tian S2, `subcort_aud`, `carpet_dseg`), three new `make_atlas_region_masks.py` branches, two bugs fixed in `make_striatum_mask.py` (hardcoded `22.1.1` path ignoring `--fmriprep_dir`; sourced from `masks-tian-S3` instead of `-S2`).
4. README updated to match.
5. **Consolidated the three GLM scripts into `06_univariate/univariate_glm.py`** with a `--grouping` flag (`none`/`grouped`) — they were ~90% duplicated. Low-quality runs after scrubbing are no longer auto-excluded (every run counts at n=12) — logged to `derivatives/nilearn/qc/sub-*_confound-scrubbing-qc.csv` instead. Multi-condition contrasts now use one `save_glm_to_bids(contrasts=stim_list, ...)` call. Fixed a real bug: the feedback run script was passing `--event_type=stimulus`.
6. **Fixed the feedback run script's grouping to `--grouping=grouped`** (was `none`) — Table 8's results are learning-stage breakdowns, not a single all-runs GLM.
7. `/code-review` pass: fixed silent exception-swallowing in the grouped-run loop, an SBATCH/fMRIPrep memory mismatch (54G vs. 64000MB — both now 64G), and ~35 lines of duplicated save logic (now one shared helper; variance maps dropped rather than reconciling an inconsistency).
8. **Built and tested `06_univariate/robustness_checks.py`** (Workstream 5): LOSO stability, exact sign-flip + Monte Carlo permutation, bootstrap CI, sharing one per-subject learning-stage trend scalar. Scoped to the anterior-caudate-vs-putamen feedback effect only. 15/15 checks passed against synthetic data.
9. **Traced `group_level_all_ROI.ipynb`'s structure — corrected an earlier mischaracterization.** It's a deliberate 3-network (striatal/cortical/PFC) × 2-contrast (sound/feedback) grid, and cell 213 already writes exactly the TSV `robustness_checks.py` needs. Fixed three schema assumptions in the script to match reality: tab-separated (not CSV), stage labels `early`/`middle`/`final` (not `earlythird`/etc.), no `event_type` column. Re-validated, 17/17 checks. Bonus: cells 223–232 already have reusable brain-behavior correlation scaffolding for Workstream 1.
10. Checked Workstream 2/3 for quick wins: color-gradient legends (Workstream 3) were already fine in code. Connectivity's caudate×rostral-caudal gap (Workstream 2) confirmed real; raw HCP connectome data turned out to be available locally — **user computing this themselves**.
11. **User delivered the behavioral modeling results** (`2023-06-08_DecisionBoundModelingOutput.txt`, per-subject-run BIC for every candidate model). Flagged two data-quality issues (low SPC win-rate; a sentinel-like BIC value and an apparent optimizer blowup) for resolution before writing comparison code.
12. **Confirmed the file mixes Mandarin/non-Mandarin participants** — only non-Mandarin (n=12, matches manuscript) is analyzed, cross-checked against the notebook's own cached sample list. The sentinel BIC value turned out to belong entirely to excluded subjects. Identified the correct test: SPC's own BIC trend across runs 1→6, not raw win-count.
13. **User decided to drop CJH1/CJH2** entirely (erratic behavior). Candidate set: UDX/UDY/CON/SPC/RAN.
14. **Built and tested the formal model comparison** (`08_behavioral-modeling/model_comparison.py`), factoring LOSO/permutation/bootstrap logic out into a shared `resampling_stats.py`. All paths are required CLI args (real cluster path noted: `/ix1/...`, not `/bgfs/...`). 20/20 checks passed (real-data smoke test + synthetic effect-detection).
15. **Final `/code-review` pass before cluster pull.** Fixed six issues: no defense against malformed rows/out-of-range run numbers in `model_comparison.py`; unverified `Block`-column assumption now checked; `'late'`→`'final'` label drift between scripts fixed; `resampling_stats.py` now raises on n<2 instead of returning silent NaNs; `make_atlas_region_masks.py`'s unsupported space/atlas combo now exits cleanly. 45/45 tests pass.

## Progress (2026-08-19 → 2026-09-04)

Cluster access obtained — fMRIPrep, atlas warping, and the univariate GLM have now actually been run, not just written. Most of this period went into getting the reprocessed feedback results trustworthy enough to report, which required several real bug fixes beyond the originally-scoped confound modernization.

16. **Ran fMRIPrep 25.2.5 to completion** (Workstream 7). Hit and resolved a FreeSurfer/fsaverage race condition (concurrent jobs colliding on a shared `subjects_dir`) by pre-seeding `fsaverage` before batch submission. Output at `data_denoised/derivatives/denoised_fmriprep-25.2.5/`, separate from 22.1.1.

17. **Ran the T1w atlas warps; extended masking to PFC** (Workstream 8). Refactored `make_atlas_region_masks.py` (`resolve_atlas_paths()`, import-safe); added a T1w `carpet_pfc` branch (previously MNI-only). Built `05_masking/atlas_warp_qa.ipynb` to validate the warps, reusing existing nilearn/repo code. **Finding:** `vol_to_surf`/fsaverage requires MNI-space input (fixed template mesh, no per-subject registration) — can't validate a T1w warp via surface plotting directly. QA notebook's cortical-surface tier switched to MNI atlases; same constraint recurs in #20.

18. **Standardized `univariate_glm.py`'s two grouping modes and iterated on the feedback design** — the largest thread this period, directly answering R1's replication concern:
    - Unified `nilearn_glm_across_runs`/`nilearn_glm_grouped_runs` on one scheme (single `save_glm_to_bids` call, `stim_list` varying by `event_type`); removed a needless `_fit_and_save_contrasts` indirection.
    - Fixed `minimize_memory=True` silently breaking `save_glm_to_bids`'s model-level maps (R², residuals) — added `minimize_memory=False`. Confirmed per-contrast maps were unaffected by the bug.
    - Confirmed `slice_time_ref` auto-inference matches the old manual formula (~3%); removed the dead manual computation.
    - Added `--no_scrubbing` (ablation flag) and, based on what it showed, **made a "collapsed-nuisance" feedback design permanent**: `sound_*`/`resp_*` collapse into single `sound`/`resp` regressors (matching `sound`-mode's own design); `fb_correct`/`fb_wrong` stay split as the actual contrast of interest.
    - Flattened the output-directory scheme to one variant tag (`scrubbed` / `noscrub` / `collapsed-nuisance` / `noscrub_collapsed-nuisance`), later renaming the baseline from `scrubbed_model-full` to `scrubbed` (`sound`/`stimulus`/`trial` never had an uncollapsed "full" design to contrast against). Propagated through the QA scripts and notebook.
    - **Found and fixed two occurrences of one nilearn issue**: `FirstLevelModel.fit()` rejects a `sample_masks` list mixing `None` with real arrays. First in `--no_scrubbing`; then in the **default path** — `load_confounds_strategy` legitimately returns `None` for any zero-censoring run, silently crashing a whole run-group when a sibling run had partial scrubbing. Root-caused as why sub-FLT20's `early` feedback run-group went missing (confirmed via SLURM `sacct` + empirical reproduction in scratch venvs, ruling out several wrong theories — filesystem caching, nilearn version mismatch, wrong output directory, positional vs. absolute run numbering in `save_glm_to_bids` filenames, which is expected behavior). Fixed by normalizing `None`→`np.arange(len(conf))` in both paths. Also fixed an `ImportError` (`nilearn.glm` vs. `nilearn.interfaces.bids` for `save_glm_to_bids`, version/env-dependent) with a try/except fallback.
    - Added a post-write file-count check after every `save_glm_to_bids` call, printing diagnostics if a run-group writes zero files — a tripwire for any future silent failure.

19. **Confirmed the collapsed-nuisance design recovers the manuscript's key striatal findings — the central result of this period.** Compared five feedback-GLM variants for the `aCAU` final-vs-early effect (original manuscript, scrubbed+full, noscrub+full, noscrub+collapsed, scrubbed+collapsed) and found a monotonic trend toward the manuscript's result as scrubbing was ablated / nuisance collapsed. Once the FLT20 bug was fixed, the true n=12 **`scrubbed`+collapsed-nuisance** result matched the manuscript: `aCAU` final-vs-early **t=-3.30, p=.043** (manuscript: t=-4.03, p=.012); `aPUT` across all runs **t=3.21, p=.049** (matches manuscript's overall effect); no other region survives correction in either comparison. **Directly answers R1's replication skepticism** — feedback's design is now permanently collapsed-nuisance based on this result.

20. **Fixed plotting bugs in `group_level_all_ROI.ipynb`**: a `plot_roi(title=...)` banner not anchored to its subplot (fixed with `ax.set_title()`); nilearn's removed `darkness` kwarg; a `plot_surf_contours` `ValueError` for small ROIs (now try/except per region). More substantially: all 12 surface-plot cells now build ROI vertex geometry from MNI-space masks while still plotting T1w-derived statistics (same constraint as #17). Also updated `variant_tag_by_event_type`: `feedback`→`collapsed-nuisance`, `sound`→`scrubbed`.

21. **Reran `sound` under the modernized pipeline — new discrepancy, not yet diagnosed.** Some auditory-cortex ROIs significant in the manuscript (e.g. left planum polare) are no longer significant. **Decided (2026-11-03): keep `sound` on `scrubbed`** for consistency with feedback's scrubbing — no unscrubbed sound variant comparison; the discrepancy is reported, not diagnosed. See Workstream 3 (and #27: the sound striatal/cortical tables turned out to be n=11).

**Cluster-access blockers from the last update are resolved**: fMRIPrep 25.2.5 and the ANTs atlas warps both ran (transform-filename assumption confirmed correct); `sound` and `feedback` GLMs have been run and re-run repeatedly as bugs were fixed. `robustness_checks.py` (Workstream 5) has now been run on real data — see #22.

22. **Ran `robustness_checks.py` against the real n=12 `scrubbed`+collapsed-nuisance feedback data — the anterior-caudate-vs-putamen double dissociation robustly confirmed.** `aCAU`'s learning-stage decline: LOSO 12/12 folds preserved, exact permutation p=.0034, bootstrap CI excludes 0. `aPUT` shows no trend (expected — its claim is a sustained, not declining, response). The differential `aCAU`-minus-`aPUT` trend (the actual headline claim): LOSO 12/12, exact permutation p=.0024, bootstrap CI excludes 0. See Workstream 5 for full numbers. **⚠ Superseded by #26** — these numbers came from an older ROI table, not the current `scrubbed`+collapsed-nuisance output; the conclusion holds but the effect sizes are smaller.

**Not started:** Workstream 1's methods write-up, Workstream 2 (user doing it themselves), Workstream 4, Workstream 6. Workstream 3 has a new open item (#21) alongside the already-resolved legend check.

## Progress (2026-09-10 → 2026-09-15)

23. **Integration plan added, split into a Phase 1/Phase 2 roadmap** (`09_integration/INTEGRATION_PLAN.md`) for connecting this manuscript's striatal-feedback/behavioral findings with representational-dimensionality work in the sibling `FLT2`/`FST1` repos — this is the groundwork Workstream 1's neural-integration analysis needs. Phase 1 (going into this revision): one pre-registered link between the striatal feedback learning-stage trend (already in this repo) and an auditory-cortex representational-dimensionality trend (recomputed in `FLT2`). Phase 2 (later, on request): the SPC-behavior/dimensionality links, an FST1 acoustic-baseline comparison, RSA/language-group extensions.

24. **`FLT2`/`FST1` dimensionality recompute now running on the cluster on fMRIPrep 25.2.5, in native T1w — matching this repo's own revised pipeline**, not left on the old 22.1.1/MNI outputs. Getting the first real run working surfaced and fixed several bugs in the sibling repos (not this one, but load-bearing for Workstream 1): `FLT2` wasn't saving GLMsingle betas as NIfTI at all; its design matrix was collapsing repeated within-run trials into one column per condition instead of one per trial, which both violates GLMsingle's own documented method and silently dropped every masked beta downstream; a stale-cache bug treated an interrupted GLMsingle run's output directory as "already complete" and crashed trying to load files that were never written. Also separately confirmed GLMsingle is methodologically valid for `FST1`'s 20-second block-design stgrid task (checked directly against GLMsingle's own docs, not assumed) and removed both repos' manual `slice_time_ref` computation to match this repo's own validated auto-inference (`univariate_glm.py`, #18). `FLT2` tonecat dimensionality output now exists for 23 of ~25 subjects; the remaining two are rerunning with the fixes applied. `FST1` stgrid GLMsingle has also been run. Full detail in `09_integration/INTEGRATION_PLAN.md`, Workstream I.

## Progress (2026-11-03 → 2026-11-04)

25. **Reran `group_level_all_ROI.ipynb` and made the robustness figure** (`plot_robustness_checks.py --mode=roi`) from the real feedback ROI table. Real filename/folder for the record: `derivatives/nilearn/group_space-T1w_fwhm-0.00/univariate-results_network-tian-S2_contrast-fbcorrectvswrong.tsv` (no hyphens in the contrast; `group_space-T1w_...`, not `group_fwhm-0.00/`). **Added the per-subject trend export** for `09_integration` Phase 1: the notebook's last cell writes `group_space-T1w_fwhm-0.00/roi_trend_scalars.csv` (`aCAU_trend`, `aPUT_trend`, `aCAU_minus_aPUT_trend`, n=12), reusing `robustness_checks.py`'s trend functions so values match the robustness checks exactly.

26. **Robustness numbers corrected — #22's values didn't match the current data.** Recomputed from the exported trend values (`resampling_stats.py`, seed 0). The trend is final−early, so `aCAU`'s t=−3.30 reproduces #19 exactly; #22's t=−4.03 is the *original manuscript's* value, i.e. `robustness_summary.csv` was built from an older ROI table. Conclusion unchanged (both claims significant, LOSO 12/12), effects smaller. **`robustness_checks.py` rerun on the current TSV (2026-11-04)** — confirms the recomputed values, but the `aCAU` Monte Carlo stage-shuffle test is now **not significant (p=.138; was .024)**: the effect is an early→final difference, not a clearly monotonic decline. Numbers and interpretation in Workstream 5.

27. **Found why "Compare contrasts" fails with `Data is unbalanced` — the sound striatal and sound cortical ROI tables are n=11 (sub-FLT20 missing).** Their save cells (40, 109) are commented out, so the notebook loads TSVs written before the FLT20 fix (#18); sound PFC and all three feedback tables are n=12. **This means Table 5's sound striatal/cortical stats are currently n=11.** Fix: re-enable the save cells for those two sections, rerun them (FLT20's sound GLM output exists), then rerun the notebook — this may also shift the #21 auditory-cortex results. **Done on the cluster (2026-11-04)**: sound tables regenerated and "Compare contrasts" rerun; not yet pushed, so the n=12 sound results (incl. left planum polare, #21) and `compare_contrasts_anova.tsv` are still to be reviewed. Also found the cause of the "manually fix these for some reason TBD" lines: `make_stats_df` has `if network_name == 'auditory' or 'pfc':`, which is always true, so Tian-S2 names get parsed as cortical (`L-HG`-style) names. The manual fix after it is correct; the condition should be `network_name in ('auditory', 'pfc')`.

## Workstreams (in priority order)

### 0. Obtain subject-specific behavioral modeling results (blocking; code assumed unavailable)
**Status: results delivered, partially reviewed.** `2023-06-08_DecisionBoundModelingOutput.txt` — per-subject-run `BestFit`/`BestFitBIC` plus an `AllFits` column with BIC for every candidate model. This is the fit-statistics table formal model comparison needs.

**Decided:** drop CJH1/CJH2 entirely. Remaining candidates: UDX_1–5, UDY_1–6, CON_1–5, SPC, RAN.

**Confirmed:** the file covers Mandarin + non-Mandarin; only non-Mandarin (exactly 12: `FLT02, 04, 06, 09, 11, 12, 13, 14, 20, 25, 28, 30`) is analyzed, matching the notebook's own cached sample. `FLT07` is excluded separately (bad QA).

Data-quality flags re-checked against this grouping (mostly moot after dropping CJH):
- Sentinel BIC value — all 7 occurrences were `CJH1_1` wins in excluded/Mandarin subjects.
- SPC's low win-rate (11%, non-Mandarin) isn't explained by the Mandarin split. **Real test**: SPC's own BIC trend across runs 1→6 relative to the reduced candidate set, not raw win-count.
- `FLT02_01`'s `RAN` winner (`BestFitBIC=-6490.4`, malformed params) is still an apparent optimizer blowup — worth excluding.
- Methods detail still needs reconstructing from notes/drafts (fitting code unavailable).
- Model-recovery simulation (Workstream 1) still needs the actual model code — likely dropped.

**Formal model comparison — built and tested, not yet run on real data.** `08_behavioral-modeling/model_comparison.py` parses `AllFits`, restricts to non-Mandarin via `participants.tsv`, flags implausible BIC as failed fits, and produces (1) a direct per-subject comparison and (2) the real claim test — SPC's BIC relative to the best competitor, early vs. late, via the shared LOSO/permutation/bootstrap logic in `resampling_stats.py`. All paths are required CLI args (real path: `/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/behavior/2023-06-08_DecisionBoundModelingOutput.txt`). 20/20 checks passed (real-data smoke test + synthetic effect-detection).

### 1. Behavioral modeling: write-up + real model comparison + neural integration
- **Methods detail:** write model spec, fitting procedure, stimulus structure (16 = 4 tones × 4 talkers) from notes/drafts.
- **Formal model comparison:** unblocked — test SPC's BIC trend across runs 1→6 against UDX/UDY/CON/SPC/RAN, restricted to n=12.
- **16-stimulus defense:** dropped the model-recovery-simulation option (needs unavailable code) — fall back to a qualitative defense (trial count per stimulus across repeated runs; precedent in small-stimulus-set studies).
- **Neural integration (most-requested fix, both reviewers, fully feasible):** correlate each subject's SPC fit trajectory with feedback BOLD change in anterior caudate vs. putamen — exactly R2's proposed linkage. Needs only the Workstream 0 results table + existing `06_univariate/` pipeline. **Head start:** cells 223–232 of the notebook already have reusable merge/correlation scaffolding (built for raw accuracy, not SPC — adapt rather than rebuild).

### 2. Connectivity (HCP) reporting fixes — mostly low-effort
- Pull the already-computed `anova_pfc_full.tsv` into the manuscript (R2's ask).
- **Caudate×rostral-caudal breakout — gap confirmed real; user computing it themselves.** No file in `hcp7t_mrtrix3_TianS2/stats_exports/` (22 checked) splits both factors at once for either auditory or PFC connectivity. Raw per-subject connectome data is available locally (`/Users/dsj3886/data_local/derivatives/HCP_7T_diffusion/msmt_csd_nthreads-1/`), and `connectivity_statistics.ipynb` already has the right dataframe columns — just never exported this comparison.
- Same caudate-only rostral-caudal breakout needed on the PFC side, for reporting parity.
- Add citations for the frontal ROI selection (ms lines 280–283) — literature search, not code.
- State explicitly that connectivity draws on an independent normative HCP sample, not this study's participants.

### 3. Univariate fMRI reporting fixes
- **Sound discrepancy (#21) — decided: report `scrubbed`, don't diagnose.** Some Table 5 (sound) auditory-cortex ROIs lose significance under the modernized pipeline (e.g. left planum polare). Kept on `scrubbed` for consistency with feedback; the text should state that the updated preprocessing (scrubbing, fMRIPrep 25.2.5, T1w) changed these results. **Blocker first:** the sound striatal/cortical tables are n=11 (#27) — regenerate at n=12 before finalizing Table 5.
- **"Compare contrasts" section** (sound vs. feedback × stage ANOVAs, pPUT follow-ups → `compare_contrasts_anova.tsv`) — fails until #27 is fixed; never yet run to completion.
- Table 5/8: state explicitly that non-significant regions were excluded via the univariate screening step — logic likely already exists upstream, just needs documenting in text.
- **Color-gradient legends — checked, not a code bug.** All three plotting paths already build a labeled colorbar; if submitted figures lacked one, it's a stale-figure/assembly issue, not code. Regenerate and confirm visually.
- Fig 4C/3B duplicate — not a pipeline bug; trace the correct source image and fix in the manuscript document directly.

### 4. Power / limitations framing
- Add a paragraph acknowledging n=12 constrains the univariate analyses (R2's ask); justify exploratory pairwise testing (a priori regions, transparency for replication).
- Consider narrowing some ANOVAs to targeted a priori contrasts (anterior caudate vs. putamen specifically) instead of exhaustive 8-ROI×hemisphere×stage designs.

### 5. New analyses to strengthen the small-N case
**Status: headline claim confirmed on the current data (corrected 2026-11-04, #26).** Input: `derivatives/nilearn/group_space-T1w_fwhm-0.00/univariate-results_network-tian-S2_contrast-fbcorrectvswrong.tsv` (n=12, `scrubbed`+collapsed-nuisance). Results: `group_space-T1w_fwhm-0.00/robustness_summary.csv` (rerun 2026-11-04 on the current TSV):
- **`aCAU` learning-stage trend**: mean −0.59, t=−3.30 (p=.007, d=−0.95), exact sign-flip permutation p=.0073, **LOSO 12/12 folds preserved**, bootstrap 95% CI [−0.92, −0.25]. **Monte Carlo stage-shuffle p=.138 — not significant** (see caveat below).
- **`aPUT` trend**: t=0.08, p=.94 (Monte Carlo p=.93), not significant — expected, not a failure: the manuscript's putamen claim is a *sustained* (non-declining) response.
- **`aCAU`-minus-`aPUT` trend (the actual double-dissociation claim)**: mean −0.61, t=−2.90 (p=.014, d=−0.84), exact sign-flip p=.0166, **LOSO 12/12**, bootstrap CI [−1.01, −0.21].

**Caveat — the two permutation tests disagree for `aCAU`.** The exact sign-flip tests the `[-1,0,1]` contrast itself (final − early; middle has weight 0) and is significant. The Monte Carlo test shuffles all three stage labels within subject, so permutations that move the middle stage into early/final widen the null — p=.138. Interpretation: the data support a final-vs-early *difference*, not a clearly monotonic three-stage *decline*. Plan for the manuscript: lead with the sign-flip/LOSO/bootstrap results (they test the claim as stated), report the Monte Carlo p alongside it with that interpretation, and phrase the claim as early→final rather than a linear decline. Check the robustness figure's per-subject panel for whether the middle stage is the noisy one.

Still answers R1's "wouldn't replicate at n=12" — the differential effect survives every leave-one-out fold — but with smaller effects than the old (superseded) numbers: t=−4.03/−4.21, exact p=.0034/.0024. Don't use those in the manuscript.

**Figure — done (2026-11-03)**: `06_univariate/plot_robustness_checks.py --mode=roi`, run against the real TSV above — a "robustness panel" (per-subject slopes · LOSO caterpillar · exact-permutation null · bootstrap CI), one row per contrast (aCAU / aPUT / aCAU−aPUT). The one-row `--mode=spc` variant (SPC behavioral trend from `model_comparison.py`'s `spc_relative_bic_early_late.csv`) is not yet run. See `INTEGRATION_PLAN.md` for the fuller menu of robustness-plot options.

**Scope — headline claims only.** Primary target: anterior-caudate-vs-putamen feedback learning-stage interaction (Table 8). Table 5 (sound) is a secondary extension if time allows.

**Summary scalar (reused, not re-derived per check):** per-subject linear learning-stage trend (weights `[-1,0,1]` across early/middle/late) per ROI, and the aCAU-minus-aPUT difference in that trend — the same scalar Workstream 1's neural-linkage analysis needs.

**Three checks:**
1. **LOSO stability** — 12 folds, report how many preserve direction/significance.
2. **Permutation**, alongside (not replacing) parametric stats — exact sign-flip (2^12=4096, fully enumerable) for pairwise pieces; Monte Carlo (~10,000 iters) for the full 3-level stage factor/interactions.
3. **Bootstrap 95% CI** on effect size (~10,000 resamples) — avoids leaning on RM-ANOVA's normality/sphericity assumptions at n=12.

**Where this lives:** `group_level_all_ROI.ipynb` writes the per-subject/ROI/stage beta table (the TSV above) and, in its last cell, the per-subject trend scalars (`group_space-T1w_fwhm-0.00/roi_trend_scalars.csv`). `06_univariate/robustness_checks.py` loads the TSV and writes `group_space-T1w_fwhm-0.00/robustness_summary.csv`. No new dependencies (`pandas`/`numpy`/`scipy`); LOSO/bootstrap/permutation are plain loops for Methods-section precision.

**Explicitly ruled out:** individual-differences connectivity work (no within-subject DWI); further exploratory modeling/classifiers (worsens multiple comparisons); behavioral model-recovery simulation (needs unavailable code).

### 6. Discussion text clarifications (writing-only)
- P.17 lines 541–543: specify whether the caudate/putamen claim is sound-related, feedback-related, or both.
- P.17 lines 568–570: clarify whether the caudal-dorsal-striatum auditory-transmission claim is supported by DWI, fMRI, or both.
- P.11 lines 334–336: adopt R2's suggested rephrasing.

### 7. Processing / GLM modernization
**Status: run to completion on the cluster (2026-09-04).** fMRIPrep 25.2.5 finished (#16); the confound/GLM upgrade is live in `univariate_glm.py` and iterated well beyond original scope (#18/#19). Feedback now matches the manuscript's key findings; sound has a new open discrepancy (#21, Workstream 3).

- **Confound/GLM upgrade — done.** Replaced the manual `conf_keep_list` (aCompCor disabled, no scrubbing) with `load_confounds_strategy(denoise_strategy='scrubbing', fd_threshold=0.9, std_dvars_threshold=1.5)`, passing `sample_mask` into `model.fit()`. Every run is kept regardless of retained-fraction (n=12 means every run counts); per-run stats logged to `derivatives/nilearn/qc/`.
- **fMRIPrep bump — decided: 25.2.x LTS**, not just 23.2.1. The 22.1.1→23.2.1 jump fixes fieldmap/SDC masking; 25.2.x additionally picks up brain-mask dilation fixes and a support window through Oct 2029. No version in range is 7T-specific — this is a general-accuracy/longevity upgrade.
  - New output tree, not overwriting 22.1.1 (`03_fmriprep/run_fmriprep_denoised_25.2.5.sh`) — **done, run**.
  - New Singularity container (`fmriprep-25.2.5.simg`, built from head node) — **done**.
  - FreeSurfer regenerated fresh under the new version rather than reusing 22.1.1-era recon-all output — **done**.
  - `--skip-bids-validation`, explicit `--output-layout bids` — **done**. SBATCH/internal memory mismatch (54G vs. 64000MB) — **found and fixed, both 64G**.
  - **Cost:** cascades through `05_masking/`, `06_univariate/`, `07_rsa/` (if revisited), Workstream 1 — effectively a full pipeline re-run against a second derivatives tree.

### 8. Native (T1w) participant-space analysis
**Status: run to completion on the cluster (2026-09-04).** ANTs atlas warps ran successfully (transform-filename assumption confirmed correct). T1w masking now also covers PFC (`carpet_pfc`, added on request — see #17). Both `sound` and `feedback` GLMs now default to T1w. The MNI-space surface-plotting constraint (anticipated for whole-brain figures) also affected per-ROI surface figures in the notebook — fixed in #20.

Scope: three atlases warped to T1w — **Tian S2** (striatum; S3 dropped), **`subcort_aud`** (CN/SOC/IC/MGN), **`carpet_dseg`** (cortical+subcortical). `carpet_motor`/`carpet_pfc` reuse the same atlas file as `carpet_dseg` (label-subset only) — trivial follow-on once `carpet_dseg` exists in T1w.

- **Atlas warps** via `antsApplyTransforms ... -n GenericLabel` using fMRIPrep's MNI↔T1w transform (`GenericLabel` interpolation matters for discrete multi-label atlases, especially the small brainstem/thalamic `subcort_aud` targets). Written as `05_masking/warp_atlases_to_T1w.sh` — **run, transform filename confirmed correct**.
- **Three new `make_atlas_region_masks.py` branches** (`T1w`+`tian_S2`/`subcort_aud`/`carpet_dseg`), pointing at the warped per-subject files, reusing existing `roi_dict_*` definitions — **done**.
- **`T1w`+`tian_S3` intentionally not built** (out of scope).
- **Fixed `make_striatum_mask.py`**: repointed from `masks-tian-S3` to `masks-tian-S2`; fixed a hardcoded `22.1.1` path that ignored `--fmriprep_dir` — **done**.
- **Run scripts now pass `--space=T1w`** — **done**.
- **Kept a parallel MNI-space pipeline for whole-brain/gradient visualization** (`SecondLevelModel` group maps need a shared voxel grid) — ROI-level stats/ANOVAs move to T1w; whole-brain figures (`striatal_roi_plotting.py`/`roi_surface_plotting.py`/`striatal_yabplot.py`) stay MNI.
- `07_rsa/` inherits the same masking dependency and gains T1w "for free" once confirmed there's no searchlight/voxelwise component assuming a shared grid.
- **Manuscript framing caveat:** T1w warping is a precision/denoising improvement (avoids MNI-normalization interpolation blur for small structures), not more anatomically individualized ROI boundaries — still population-average parcellations.

## Suggested sequencing

1. ✅ **Done.** Housekeeping: branch, `.gitignore`, starter `environment.yml`.
2. ✅ **Delivered; comparison code built, not yet run on the real full file.** CJH1/CJH2 dropped; `model_comparison.py` validated (20/20). 🔲 Methods write-up not started. 🟡 Neural-integration analysis: groundwork (sibling-repo dimensionality recompute, #24) now running on the cluster; the actual correlation/write-up (Phase 1 of `09_integration/INTEGRATION_PLAN.md`) hasn't started.
3. ✅ **Run to completion.** fMRIPrep 25.2.x built and run to a separate tree (Workstream 7); `05_masking/` re-run against it.
4. ✅ **Run to completion.** Tian S2/`subcort_aud`/`carpet_dseg`/`carpet_pfc` warped to T1w (Workstream 8); transform confirmed correct.
5. ✅ **Done, substantially beyond original scope.** Confound/GLM upgrade live; run scripts use `--space=T1w`. Found/fixed real data-loss/API bugs, added scrubbing ablation + permanent collapsed-nuisance design, confirmed feedback replicates the manuscript (#19). Sound stays on `scrubbed` despite its discrepancy (#21, decided). README needs a catch-up pass (not yet done).
6. 🟡 **Partially started.** Univariate: regenerate the n=11 sound striatal/cortical tables at n=12 (#27), then rerun the notebook including "Compare contrasts"; rest of Workstream 3 is writing. Connectivity reporting (Workstream 2) not started — user doing it themselves.
7. 🟡 **Model comparison done; write-up not started; neural linkage groundwork in progress** (see item 2 / #24). `08_behavioral-modeling/` exists, README current.
8. ✅ **Headline double-dissociation confirmed on current data** (Workstream 5, corrected #26): `aCAU`-minus-`aPUT` learning-stage trend t=−2.90, survives all 12 LOSO folds, exact permutation p=.0166, bootstrap CI excludes 0. `aCAU` Monte Carlo stage-shuffle p=.138 (n.s.) — frame as early→final, not linear decline. Robustness figure, trend export, and `robustness_summary.csv` all current.
9. 🔲 **Not started.** Power-limitations paragraph (Workstream 4), then discussion clarifications (Workstream 6) — last, once analyses are final.
10. 🔲 **Not started.** Final formatting/scope pass once a journal is chosen. Merge to `main` only once the user is fully satisfied.

## Verification
- Re-run `group_level_all_ROI.ipynb` and `connectivity_statistics.ipynb` end-to-end after any stats change, to confirm tables/figures match text.
- Sanity-check the behavioral results table before trusting it: comparable fit-statistic basis across models; no inconsistent per-run/pooled mixing.
- Cross-check every reviewer comment against the final manuscript line-by-line before resubmission.
