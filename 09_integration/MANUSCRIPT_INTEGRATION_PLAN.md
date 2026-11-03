# FLT manuscript update: integrating FLT2, FST1 and auditory_category_learning

*Drafted 2026-10-30. Builds on `INTEGRATION_PLAN.md` (last updated 2026-09-15) and `../REVISION_PLAN.md`;
this file is the manuscript-facing layer: what goes where in the paper, what changed since 9/15,
and the ordered work to get there. Manuscript text itself was not available, so section names
below follow the structure described in `REVISION_PLAN.md`.*

## 0. Repo roles (as read on 2026-10-30)

| Repo | Role in the paper | State |
|---|---|---|
| `fMRI_auditory-category-learning` (branch `manuscript-revision`) | **Home of the manuscript.** Striatal sound/feedback GLMs, robustness checks, SPC model comparison, masks | Last commit 9/10. `REVISION_PLAN.md` modified and `09_integration/` **untracked** -- nothing in this folder is committed yet |
| `auditory_category_learning` (`sitek/`, original pipeline) | Source material only | Unique content not in the manuscript repo: Mandarin + combined behavior summaries (8/4), RSA outlier/NaN-masking fix in `rsa/rsa_roi.py` (7/28), decoding, stgrid RSA |
| `FLT2` (branch `representational-dimensionality`, unmerged) | **New results section**: representational dimensionality (PR) of tonecat responses across learning | GLMsingle → T1w masks → PR done on fMRIPrep 25.2.5 / T1w for 23 subjects; group notebook re-run 9/15 |
| `FST1` | Optional supplement: acoustic (spectrotemporal grid) baseline in the same people | GLMsingle code fixed 9/18; RD notebook adapted but **not executed**; spectrotemporal tuning script still MNI / old 22.1.1 output dir |

Assumption: "auditory category learning" = `sitek/auditory_category_learning`, mined for parts;
`fMRI_auditory-category-learning` stays the manuscript's code home.

## 1. What changed since the 9/15 plan

### 1a. FLT2 now has a real result (n=23, Mandarin + non-Mandarin, 25.2.5/T1w)
- Stage main effect on PR: F(2,44)=5.37, p=.008 (GG p=.009); stage×ROI n.s. (p_GG=.29).
- PR is **lower in final than early in every region**. Per-ROI early-vs-final, FDR across 14 ROIs:
  R-HG, R-STGa, R-ParsOp (p_FDR=.048), L-STGp, R-ParsTri (p_FDR=.050); all others n.s.
- L/R-pooled: no region survives FDR (all p_FDR ≥ .054), but bootstrap CIs exclude 0 for 6/7 regions (all but PP).
- A-priori Primary composite: early-vs-final t(22)=2.39, p=.026, d=.35; stage RM-ANOVA p=.076.

**Consequences for the paper**
1. The expected sign for analysis B is now fixed: PR **declines** with learning. With aCAU
   feedback also declining, B predicts a **positive** correlation between the two trend scalars.
2. **Phase 1's specificity check no longer works as written.** It predicted a null PR trend in the
   association composite (STGa/STGp/ParsOp/ParsTri); at group level those regions decline at least as much
   as primary (STGa d=.45, ParsOp d=.42 vs HG d=.34). Replace before running B (see §3, step E).
3. These numbers mix language groups. The manuscript sample is the 12 non-Mandarin participants; the
   stage effect must be re-established in that sample before it can carry a Results section. The
   Mandarin/non-Mandarin split does matter here (group difference in Primary PR at the middle stage,
   t=-3.38, p_FDR=.019), so the non-Mandarin-only effect may differ.

### 1b. Sample gaps
- FLT02 (in the manuscript 12) is still absent from the FLT2 outputs; FLT22 rerun status unknown.
- §9a's merge with `participants.tsv` labels only 21 of the 23 loaded subjects (10 non-Mandarin) -- at
  least one more manuscript subject is missing or unlabeled.
- `sub_list` in the notebook setup cell is stale (its exclusion comment contradicts the list; FLT06 is
  loaded but not listed). Reconcile against the 12.
- FLT06/11/14: ran cleanly; QA glance still owed.

### 1c. FLT2 notebook issues to fix before any number goes in the paper
- §13 D1 table: the "all-runs" row repeats the **final-stage** values (np²=.195) instead of §9c's
  (np²=.229). Variable reuse bug.
- Stale markdown: §8e-iii says no ROI survives FDR (five now do); §13 cites a §9c p=.003 that no longer matches.
- Not executed on the 25.2.5/T1w recompute: §9h (D_sheng cross-validation), §9i (PR ↔ accuracy),
  §11 searchlight maps, §13 D2. Searchlight D_sheng extraction loads 0 subjects (searchlight not re-run).

### 1d. Correction to the 9/15 plan: fMRIPrep did not cover both tasks in one run
`03_fmriprep/run_fmriprep_denoised_25.2.5.sh` runs `--task tonecat` only. FST1's `run_fmriprep_stgrid.sh`
(9/15) is a **second** fMRIPrep pass with `--task stgrid` writing to the **same**
`denoised_fmriprep-25.2.5/` tree. Two checks follow:
1. Did the second pass rewrite the anatomical outputs the manuscript already depends on
   (`desc-preproc_T1w`, `from-MNI152NLin2009cAsym_to-T1w` xfm, brain mask)? Compare mtimes/checksums
   against the tonecat-era files. If they changed, the T1w masks from `05_masking/warp_atlases_to_T1w.sh`
   were built on a transform that no longer matches the files on disk -- re-warp or show equivalence.
2. The tonecat-vs-stgrid affine/shape check (open since 9/14) matters more now, since the stgrid
   boldref comes from a separate run. Five-minute nibabel check per subject.
Also fix the INTEGRATION_PLAN.md line claiming one run covers both tasks.

### 1e. FST1 status
GLMsingle (T1w, block design) fixed and re-run; masking + `dimensionality_roi.py` status not
recorded in the repo; RD notebook never executed; `analyze_stgrid_spectrotemporal.py` still reads
`glmsingle_stgrid/` in MNI. Nothing FST1 is paper-ready yet.

### 1f. auditory_category_learning
- `rsa/rsa_roi.py`'s outlier-condition NaN-masking is **not** in `07_rsa/rsa_roi.py`. Only matters if RSA returns.
- `behav_conversion/FLT_group-combined_behav_summary.csv` has Mandarin + non-Mandarin behavior. Only
  matters if a language-group section is added (recommended against, §2).

## 2. Proposed manuscript architecture

Throughline (unchanged from INTEGRATION_PLAN): striatal feedback signal declines in anterior caudate
as learning proceeds **and** auditory cortical representations compress -- and the two co-vary across
people. Reviewers asked for integration, not more standalone analyses, so every addition below must
attach to that line.

| Section | Content | Source | Status |
|---|---|---|---|
| Results 1 Behavior | Accuracy/RT across stages; SPC vs UDX/UDY/CON/RAN formal comparison | manuscript repo `04_behavior`, `08_behavioral-modeling` | model comparison built; confirm run on full file |
| Results 2 Striatum | Sound + feedback univariate (T1w/25.2.5), aCAU vs aPUT dissociation + robustness panel | `06_univariate` | done; sound discrepancy (#21) open |
| **Results 3 (new)** | Auditory cortical PR declines over learning (non-Mandarin 12) | `FLT2` | needs §1b–1c first |
| **Results 4 (new)** | aCAU feedback trend ↔ auditory PR trend (analysis B) + robustness | `09_integration` | not started |
| Supplement (optional) | FST1 acoustic baseline: same-ROI PR for stgrid vs tonecat; tuning maps | `FST1` | not started |
| Out of scope | HCP structural connectivity (separate manuscript); Mandarin vs non-Mandarin; RSA; decoding | `FLT2` §9, `07_rsa`, old repo | hold for a separate paper |

Recommendations behind the table:
- Results 3 is needed even though B is the integration result: a correlation between two trends is
  hard to interpret unless the reader first sees that cortical PR changes at all. Keep it to one figure.
- Keep the language-group analyses out. They are the most interesting FLT2 result but would reopen
  the sample definition and the "drift" complaint. They are the natural core of a separate FLT2 paper.
- FST1 earns a place only if it answers a reviewer-relevant question (is the PR decline category
  learning, or generic repetition/adaptation?). The within-subject tonecat-vs-stgrid PR contrast
  does that; tuning maps do not. If it can't be ready in time, leave it out.

### Methods additions
- GLMsingle single-trial estimation (HRF library, GLMdenoise, fracridge); one column per trial;
  stated difference from the nilearn condition-level GLM used for striatum.
- PR definition, ceiling (min(n_stimuli, n_voxels); 16 stimuli in both tasks), voxel-count check (FLT2 §8d-ii).
- Stage split (runs 1-2 / 3-4 / 5-6) shared with the striatal analysis; `[-1,0,1]` trend scalar.
- ROI masks: same per-subject T1w `carpet_dseg` masks as the striatal/cortical univariate analyses.
- If FST1 is in: stgrid task, block design, GLMsingle `stimdur`, separate fMRIPrep pass (§1d).

### Figures
- New Fig (Results 3): PR by stage per region (box+strip, existing FLT2 style) + group eigenspectra by stage.
- New panel (Results 4): 12-point scatter of aCAU trend vs PR trend with LOSO-stable fit + bootstrap band;
  reuse `plot_robustness_checks.py` layout for the supplement.

## 3. Ordered work plan

**A. Housekeeping (now)**
1. Commit `09_integration/` and `REVISION_PLAN.md` on `manuscript-revision` (currently untracked/uncommitted).
2. Correct INTEGRATION_PLAN.md §Workstream I re: two fMRIPrep passes (§1d).

**B. Sample completeness** -- blocks C onward
3. Finish FLT02 (and FLT22) GLMsingle → mask → PR.
4. Confirm all 12 non-Mandarin have stage-split PR and correct group labels; fix the notebook `sub_list`.
5. QA FLT06/11/14 (tSNR, voxel counts, R² maps).

**C. Integrity checks** -- can run in parallel with B
6. Anatomical-overwrite check from the stgrid fMRIPrep pass (§1d.1).
7. tonecat vs stgrid affine/shape check (§1d.2).

**D. FLT2 group analysis on the manuscript sample**
8. Re-run §6–8e restricted to the 12 non-Mandarin; this is the Results 3 number.
9. Fix §13 D1 bug and stale markdown; run §9i (PR ↔ accuracy) on the 12.
10. Decide D_sheng: report PR as primary and ROI-level D_sheng (already in `dimensionality_roi.py` CSVs)
    as robustness. Skip the searchlight version for this paper.

**E. Re-specify analysis B, then freeze it** -- write the spec into the top of `integration_B.ipynb`
with a date before computing any correlation
11. Striatal scalar: aCAU `fb_correct − fb_wrong` `[-1,0,1]` trend (main); aCAU−aPUT differential (secondary).
12. Cortical scalar: Primary composite (HG+PT+PP, L/R pooled) PR trend (main, as before).
13. Replace the association "expected null" check with a specificity test that the data haven't
    already contradicted. Options: (a) a non-auditory control ROI from `carpet_dseg` (needs a
    region with adequate voxels; run it through the same GLMsingle/PR path), or (b) rely on the
    striatal side for specificity -- aPUT trend should *not* correlate with PR. (b) costs nothing new.
14. Predicted sign: positive r. Test: Pearson + Spearman, bootstrap CI, LOSO, exact sign-flip
    permutation via `resampling_stats.py`.

**F. Merge layer + run** (INTEGRATION_PLAN Workstream II, unchanged)
15. `subjects.csv`, `roi_crosswalk.csv`, striatal trend exporter, `integration_B.ipynb`.
16. Run B; produce scatter panel + stats table (`stats_fmt.py`).

**G. FST1 go/no-go** (decide after D; only if time allows)
17. Run stgrid masking + `dimensionality_roi.py` for the 12; execute the FST1 RD notebook.
18. Pre-specify the tonecat-vs-stgrid PR contrast per ROI. Match trial counts first: tonecat has
    ~3 reps × 6 runs per stimulus, stgrid 1 block per run, and PR is sensitive to noise level.
19. Port `analyze_stgrid_spectrotemporal.py` to the 25.2.5/T1w GLMsingle dir only if tuning maps are used.

**H. Writing**
20. Methods (GLMsingle, PR, stage split, masks, estimator difference).
21. Results 3 + 4; Discussion paragraph on striatal teaching signal ↔ cortical compression,
    framed as co-variation (n=12, correlational).
22. Response-to-reviewers mapping: R2 integration request → Results 4; R1 replication → robustness panel.
23. Merge FLT2 `representational-dimensionality` branch to `main` once the numbers are frozen.

## 4. Decisions needed from you
1. Confirm repo roles in §0.
2. Results 3 as its own section (recommended) vs folding PR into the B result only.
3. Specificity check for B: control ROI (a) or aPUT-null (b).
4. FST1 in a supplement, or saved for the FLT2/FST paper.
5. Confirm Mandarin vs non-Mandarin stays out of this manuscript.
