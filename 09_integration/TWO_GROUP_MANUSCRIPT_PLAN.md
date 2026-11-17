# Two-group FLT manuscript plan: learners (non-Mandarin) vs native listeners (Mandarin)

*Drafted 2026-10-30; updated 2026-11-04 with the first two-group striatal and cortical ROI results
(REVISION_PLAN.md #28–#33), and 2026-11-11 with the full within-group analysis for each group
(§0c; REVISION_PLAN #34–#35), and 2026-11-12 with the pPUT sound-trend and group × hemisphere
tests (REVISION_PLAN #36), and 2026-11-13: FLT15 kept, behavior rescored (REVISION_PLAN #39), and
2026-11-17: what the aCAU−aPUT group difference is made of, FMC naming, figure status
(REVISION_PLAN #40).
**This is the only live manuscript plan (2026-11-17).** The live parts of
`MANUSCRIPT_INTEGRATION_PLAN.md` (learners-only framing, 2026-10-30) and `INTEGRATION_PLAN.md`
(cross-repo technical plan, 2026-09) were merged in: FLT2/FST1 status and integrity checks (R5,
R6), the analysis-B spec and merge layer (R7), Methods and figure notes (§3), repo roles and
cross-repo risks (Appendix, §5). Both originals are in `09_integration/archive/` for the history
(e.g. the Workstream I bug log). `../REVISION_PLAN.md` is the progress log.*

**Framing (decided 2026-11-17): both groups, plus learners-only analyses.** The paper compares
learners and natives throughout, and keeps the analyses only learners can support -- above all
the `fbcorrectvswrong` headline. Each feedback contrast has one job:

| Role | Contrast | Groups | What it shows |
|---|---|---|---|
| Headline | `fbcorrectvswrong` aCAU trend; aCAU−aPUT | learners | The original finding and its robustness (Reviewer 1: replication at n=12) |
| Specificity | `fbcorrect` aCAU−aPUT trend difference | both | The correct-feedback part of the shift is absent in natives |
| Decomposition | `fbwrong` = `fbcorrect` − `fbcorrectvswrong` | learners | Headline ≈ half less response to correct feedback, half more to wrong; natives speak only to the first half |
| Shared vs different | `sound` | both | Auditory cortex, aPUT, NAc-core in both; vmPFC (FMC) and STGp differ |

Say in Methods why the contrast changes: natives average ~1 wrong-feedback trial per run and six
have none. Don't call the `fbcorrectvswrong` decline itself "learner-specific" -- that can't be
tested. Analysis B (R7) and the robustness panel stay learners-only on `fbcorrectvswrong`.

**What changed on 2026-11-04**
- One first-level model, `scrubbed-splitfb`, now gives every contrast for both groups (`sound`,
  `fbcorrect`, `fbcorrectvswrong`; `resp` is modeled but not reported). The per-ROI tables for
  both groups come from `06_univariate/extract_roi_values.py`; the stats are in
  `06_univariate/group_stats.py` and `group_level_two_group.ipynb`.
- **The planned feedback test came out mixed.** `fbcorrect` does not reproduce the learners'
  headline aCAU decline (§2 R3), so "the aCAU decline is learner-specific" can't be claimed from
  `fbcorrect` alone. The aCAU−aPUT differential trend does differ between groups.
- `resp` is dropped from reporting. The response follows the sound by <1 s (TR=3 s), and group
  differences in response time would confound any group comparison of it.
- Fixed a pseudoreplication bug in `group_level_all_ROI.ipynb`'s stage- and hemisphere-collapsed
  t-tests (REVISION_PLAN #33). Every test in this plan uses one value per participant.

## 0. Why two groups, and the logic every analysis follows

Behavior (`auditory_category_learning/behav_conversion/FLT_group-combined_behav_summary.csv`):

| Run | Mandarin accuracy (n=12) | non-Mandarin accuracy (n=12) |
|---|---|---|
| 1 | .95 (min .78) | .38 |
| 6 | .98 (min .92) | .82 |

The stimuli are Mandarin tone categories (4 tones × 4 talkers). Mandarin speakers already have the
categories; non-Mandarin speakers learn them in the scanner. Same sounds, same feedback, same session
length -- only learning differs. So the Mandarin group is a **native-listener control**:

- An effect that reflects **learning** should change across stages in learners and be absent or
  flat in natives → **group × stage** interaction.
- An effect that reflects **auditory processing or repetition** should appear in both groups.
- An effect that reflects **category knowledge** should be present in natives from run 1 and
  emerge over runs in learners.

This turns Reviewer 1's main objection (would the striatal decline replicate, and is it learning at all?)
into a direct test, and gives FLT2's language-group results a home inside the same paper.

Working title idea: *Striatal feedback signals and auditory cortical representations distinguish
learning from native listening of speech-tone categories (7T fMRI).*

## 0b. Where the evidence stands (updated 2026-11-17)

New-pipeline ROI numbers below are from `group_level_two_group.ipynb`: hemispheres and stages
averaged per participant, FDR across regions within group × contrast × network, learners n=12,
natives n=13 (including FLT15, §1). "q" is FDR-corrected.

Provenance: **new** = fMRIPrep 25.2.5, native T1w, current code (reportable once frozen).
**old** = fMRIPrep 22.1.1, MNI, `auditory_category_learning` notebooks (preliminary; numbers
will move on rerun). Strength: ● supports the claim · ◐ suggestive / indirect · ○ no evidence yet.

### Similarities between groups

| Claim | Current evidence | Prov. / n | Strength | Missing |
|---|---|---|---|---|
| Sound drives auditory cortex in both groups | Sound > baseline: HG, PP, PT, STGa and STGp q<.05 in both groups (learners t=4.65–8.21; natives t=5.65–10.2); ParsOp in natives only (t=2.78). **But it's stronger in natives:** STGp natives > learners (d=−1.23, q=.036; q=.079 without FLT15) | new, 12/13 | ● for "both"; ◐ for the STGp difference | Equivalence tests for the other auditory ROIs; FLT15 decision |
| Sound drives dorsal striatum in both groups | Sound > baseline: aPUT (learners t=6.96, natives 7.05) and NAc-core (3.93, 2.62) q<.05 in both groups. aCAU only in learners (3.54 vs 1.02); pPUT (3.62) and NAc-shell (2.86) only in natives. **No striatal ROI differs between groups** (aCAU t=1.40, all q>.05) | new, 12/13 | ● for aPUT/NAc-core; ◐ elsewhere | TOST, since "significant in one group only" is not a group difference |
| Practice speeds responses in both groups | RT run 1→6: learners t=-3.29, p=.007; natives t=-3.05, p=.011 | behavior, 12/12 | ● | Group × run test. Implication: any neural effect that only tracks RT is not learning |
| Overall cortical dimensionality similar | FLT2 PR group main effect F(1,19)=0.85, p=.37 (but see topography below) | new, 11/10 | ◐ | Full sample; equivalence test |
| PR declines across stages | F(2,44)=5.37, p=.008, pooled across groups | new, 23 | ◐ for "shared" | **Group × stage** -- the decline could be shared (repetition) or learner-only (learning). Not yet tested |

### Differences between groups

| Claim | Current evidence | Prov. / n | Strength | Missing |
|---|---|---|---|---|
| Learners learn; natives are at ceiling | Accuracy run 1→6 (`two_group_accuracy.py`, 2026-11-13): learners .38→.82 (t(11)=6.68, p<.001); natives .945→.984 (t(12)=2.29, p=.041, nearly all run 1→2, then flat ≈.98: task familiarisation). **Group × run** F(5,115)=19.06, p<.001, ηp²=.45; group F(1,23)=38.9; run F=25.9, p_GG=.0001 | behavior, 12/13 | ● | Rest the claim on the interaction, not on "natives don't change". GG correction not reported for the interaction (pingouin); F is large enough not to matter |
| Medial frontal cortex responds differently to sound | FMC sound: learners below baseline (−0.35), natives above (+0.40); d=−1.44, q=.013, holds without FLT15 (q=.025); also the only group main effect surviving FDR in the group × stage ANOVA (F=13.0, q=.033) | new, 12/13 | ● (strongest striatal/frontal ROI group difference) | Not predicted -- interpret cautiously; check it isn't driven by `resp` timing (RT differs by group) |
| Cortical PR topography differs | Group × ROI F(13,247)=5.63, p<.001. Learners higher PR in right auditory cortex (R-STGa p_FDR=.017, d=1.65; R-HG/R-STGp p_FDR=.055); natives higher in L-ParsOp (p_FDR=.055) | new, 11/10 | ● (strongest new-pipeline group result) | Complete sample (FLT02 etc.); FST1 baseline to say whether it's experience or trait |
| Primary vs association hierarchy reverses by group | Learners Primary > Association at middle/final (p=.04/.02 unc.); natives reverse at middle (p=.04 unc.); all p_FDR=.08. Primary-level group difference largest at middle stage (t=-3.38, p_FDR=.019) | new, 11/10 | ◐ | Single mixed model instead of per-stage tests |
| ~~Striatal response-period activity only in learners~~ | **Dropped (2026-11-04).** `resp` is in the model as a nuisance regressor but not reported: the response is <1 s after the sound at TR=3 s, and group RT differences would confound it. (The new pipeline did show a NAc-shell `resp` group effect, q=.008; not reported for these reasons.) | -- | -- | -- |
| aCAU feedback decline is learning-specific | Learners, headline `fbcorrectvswrong`: aCAU trend t=−3.30, exact p=.007, LOSO 12/12; aCAU−aPUT t=−2.90, p=.017, LOSO 12/12. **Cross-group test (`fbcorrect`):** aCAU trend learners −0.33 (p=.083), natives +0.07 (p=.70), difference t=−1.59, p=.13, LOSO 1/25 -- not significant. **aCAU−aPUT trend differs between groups:** −0.66, t=−2.52, p=.019, permutation p=.018, LOSO 25/25, 95% CI [−1.15, −0.17]. **A relative difference:** none of the four component trends is significant (learners aCAU −0.33, aPUT +0.07; natives aCAU +0.07, aPUT −0.20). Learners' aCAU starts above natives' and ends level with them (stage gaps +0.30, +0.15, −0.11; t=1.40, 0.62, −0.41) | new, 12/13 | ◐ -- the differential trend is robust, but `fbcorrect` only partly captures the headline effect (R3) | Reframe around what `fbcorrect` measures (§2 R3). Describe the convergence descriptively; it isn't tested |
| NAc-shell feedback differs by group | `fbcorrect` NAc-shell: learners −0.22, natives +0.13; d=−1.06, p=.016, q=.099 | new, 12/13 | ○ (not FDR-significant) | Exploratory only |
| Tone-category structure emerges with learning | Learners: Tone RDM fits > Talker across cortex and striatum; Tone fit rises early→late | old, learners only | ◐ for learners, ○ for groups | Natives not analyzed. Old "all regions" tests pool ROI × subject rows as independent (e.g. t=14.6), which inflates them -- redo at subject level |
| Tone decoding differs by group | Natives' striatal tone decoding near chance (old, partial); cross-group "train native, test learner" decoding started with 5 natives / 7 learners | old, partial | ○ | Not reportable as is |
| Spectrotemporal (FST1) baseline differs | Pooled group sound map (n=21) only; group-difference section never updated; no stgrid PR yet | old | ○ | Everything |

### In progress
- FLT2 T1w recompute: FLT02/FLT22 reruns; notebook fixes (§13 D1 bug, stale text, unexecuted §9h/§9i).
- FST1: stgrid GLMsingle done in T1w; second fMRIPrep pass for stgrid (check anatomical overwrite); masking and PR not confirmed.
- ~~Manuscript striatal pipeline: learners only so far; sound discrepancy (REVISION_PLAN #21) open.~~
  Done 2026-11-04: both groups are run; the sound discrepancy is resolved by the single model
  (left PP t=3.14, q=.012).
- Behavioral model comparison: built and tested; confirm it has been run on the full file.
- ~~FLT15 events check (§1)~~ done: kept (§1). ~~TOST~~ done (R2).

### Bottom line (2026-11-04)
- **Similarity claims hold on the new pipeline:** sound drives auditory cortex, aPUT and
  NAc-core in both groups. They still need equivalence tests before "no difference" can be said.
- **Group differences in ROI response levels are few and cortical/frontal:** FMC and STGp for
  sound. Nothing in caudate or putamen differs in overall level.
- **The central claim is partly supported.** Learners and natives differ in how the anterior
  caudate's correct-feedback response changes *relative to putamen* (aCAU−aPUT, robust). But
  `fbcorrect` alone doesn't reproduce the learners' headline aCAU decline, and its aCAU-only
  group difference is not significant. The headline decline is about half less response to
  correct feedback and half more response to wrong feedback; natives can only be measured on
  the first half. The framing decision at Phase 1 step 8 has to be made on this basis (§6).
- **The strongest group difference overall is still cortical:** PR topography (FLT2).
- **Wording for the feedback result (2026-11-17):** the groups differ in how aCAU changes
  *relative to* aPUT, not in either region alone. Don't write "learners' aCAU decreased while
  aPUT increased": learners' aPUT trend is ≈0 and no single-region trend is significant.

## 0c. Within-group results (2026-11-11)

From `group_level_all_ROI.ipynb` run separately for each group (learners: commit `2bf49c3`, values
unchanged by the refactor; natives: `0e616aa`, `analysis_group = 'Mandarin'`). One value per
participant throughout; q = FDR across regions within the table. **Feedback rows compare
different contrasts:** learners `fbcorrectvswrong`, natives `fbcorrect`. Learners' `fbcorrect` is
in the two-group notebook (section 2).

| | Learners (n=12) | Natives (n=13) | Reading |
|---|---|---|---|
| **Sound, striatum** (stages averaged) | aPUT 6.96, NAc-core 3.93, aCAU 3.54 (q<.01); pCAU, pPUT, NAc-shell n.s. | aPUT 7.05, pPUT 3.62, NAc-shell 2.86, NAc-core 2.62 (q<.05); **aCAU 1.02, pCAU −0.02 n.s.** | aPUT/NAc-core shared. aCAU sound response only in learners; pPUT/NAc-shell only in natives. None of these is a significant *group difference* (two-group 5b) |
| **Sound, striatum, hemisphere** | no hemisphere effect (F=0.42, p=.53) | hemisphere effect (F(1,12)=6.87, p=.022) and hemisphere × axis (p=.040); left t-values larger (aPUT-lh 9.06 vs rh 4.96; pPUT-lh 4.05 vs rh 2.38) | Left-weighted in natives only, but **not a group difference**: group × hemisphere (left − right, all striatal regions) −0.41, t=−1.25, p=.22, permutation p=.23 (two-group 5c; natives' t²=6.87 reproduces the F) |
| **Sound, striatum, stage** | pPUT declines: early vs final t=3.52, q=.029 (early t=3.06 vs baseline, then ~0.7) | pPUT sustained: early vs final t=1.03, p=.32 (early 4.31, middle 2.39, final 1.95) | **Not learner-specific.** Group test on the `[-1,0,1]` trend (two-group 4b): learners −1.41 (t=−3.52, exact p=.003, LOSO 12/12), natives −0.71 (t=−1.03, p=.32, very variable), difference −0.70, t=−0.88, p=.39, CI [−2.15, 0.81]. Both groups trend downward; reportable only as "declines in learners". No striatal stage main effect in either group |
| **Sound, auditory cortex** | HG, PP, PT, STGa, STGp q<.001; ParsOp q=.066 | Same regions, larger t (PT 10.2, STGp 9.47); ParsOp q=.020; **left PP t=3.96, q=.003** | Shared; stronger in natives (STGp group difference, two-group 5b). Table 5's left PP replicates in natives |
| **Sound, PFC** | Below baseline: FP −6.64, OFC −4.39, FMC −3.10 (q<.03); PreCG, FO positive | FP −2.97 (q=.043) only; **OFC −0.54, FMC +2.35 (q=.080)**; PreCG, FO, IFGo positive | Sound-evoked medial/orbitofrontal deactivation is **learner-specific** (FMC group difference q=.013; OFC q=.060). PFC hemisphere effect in learners only (F=11.96, p=.005) |
| **Feedback, striatum** | `fbcorrectvswrong`: aCAU decline (final − early t=−3.30, q=.043); aPUT positive (3.21, q=.049) | `fbcorrect`: aPUT positive (4.23, q=.007); **no stage change in any region** (all early-vs-final q=.90; stage F=0.17) | Natives' correct-feedback response is flat across the session |
| **Feedback, auditory cortex** | `fbcorrectvswrong`: STGp positive (3.84, q=.019) | `fbcorrect`: below baseline, strongest early (PT early t=−4.23, q=.025; HG/PP/PT stage-averaged q≈.09) | Learners' `fbcorrect` is also below baseline here (HG t=−6.15, two-group section 2): auditory cortex dips after correct feedback in both groups |
| **Feedback, PFC** | nothing survives | nothing survives | -- |
| **Sound vs feedback** (Compare contrasts) | sound > feedback in aCAU, aPUT, NAc-core and most cortex | sound > feedback in aPUT, pPUT, NAc-core, NAc-shell; **not aCAU** (F=1.50, p=.25); HG event × stage p=.040 | Mirrors the sound results above |
| **Brain–behavior** (uncorrected) | `fbcorrectvswrong` early: aCAU r=−.75, pCAU r=−.78 with accuracy | few, scattered (accuracy at ceiling) | Exploratory; natives' ceiling makes their correlations uninformative (§5 Risks) |

**What this adds to §0b**
- The clearest *new* learner-specific result is in the `sound` contrast: learners (not natives)
  show sound-evoked medial/orbitofrontal deactivation (FMC group difference q=.013). The other two
  candidates did not hold up as group differences (2026-11-12): pPUT's sound decline (group
  p=.39; natives decline too, just more variably) and the natives' left-weighted striatal sound
  response (group × hemisphere p=.22).
- Per-region trend group differences (two-group 6b, `gs.trend_difference_tests`, exploratory,
  2026-11-13): the correct-feedback (`fbcorrect`) trend differs between groups in **ParsTri**
  (d=−1.18, q=.030), **STGa** (d=−1.15, q=.030) and **ParsOp** (d=−1.00, q=.046). Learners stay
  flat or decline slightly (−0.13 to −0.24); natives increase (+0.47 to +0.76). Nothing survives
  in the striatum or PFC, and no `sound` trend differs. FDR is only within the 7 auditory regions;
  12 such families were searched. Check before interpreting: natives' auditory feedback response
  is below baseline early (§0c), so their "increase" may be recovery from an early dip rather
  than a learning effect.
- Laterality elsewhere (two-group 5c, exploratory): learners' PFC sound response is left-weighted
  (t=3.27, p=.008; natives t=1.54; group p=.088), and learners' `fbcorrect` is left-weighted in
  auditory cortex and PFC (t=2.58, 2.37). Only STGa `fbcorrect` differs between groups at p<.05
  (t=2.70, p=.013, q=.093). Nothing survives FDR.
- Natives show no stage-related change in any striatal response, sound or feedback. That fits
  "learning-related change is learner-specific", but it's absence of evidence until tested as
  group × stage (TOST or Bayes factor, §1 Power).
- FLT15 (natives' trend export): aCAU−aPUT `fbcorrect` trend −0.61, opposite to the native mean
  (+0.27). Keeping them (§1, decided 2026-11-13) makes the planned group difference more
  conservative.

## 1. Sample

| Group | Behavior | Imaging | Notes |
|---|---|---|---|
| non-Mandarin (learners) | 12: FLT02 04 06 09 11 12 13 14 20 25 28 30 | same 12 | FLT02 run-1 response coding corrupted (behavioral modeling n=11) |
| Mandarin (natives) | 13: FLT03 05 08 10 15 17 18 19 21 22 23 24 26 | same 13 | **FLT15:** kept, see below |
| Excluded | FLT07 (bad QA) | | FLT27 appears in some FLT2 lists -- confirm status |

**FLT15: kept in every analysis (decided 2026-11-13).** The old repo excluded them for "bad
responses" (`behav_conversion/behavior_analysis.ipynb`, commit `47ab055`). The events QA
(`04_behavior/qa_events_summary.py`) shows the responses are fine; the problem was formatting
from their one-off 2025 reconversion (commit `7639a4b`):
- **Keys stored as floats** (`resp_6.0`). The accuracy scripts took the key as the label's last
  character, read `'0'`, and scored every FLT15 trial wrong: 0% accuracy in every run, which is
  what the old exclusion and the native brain–behavior correlations saw. Fixed in
  `create_behav_confusion_matrices.py` (commit `f36d397`) and `behavior_analysis.ipynb`; the
  native notebook was rerun on the corrected `run_accuracies.tsv`.
- **Actual performance:** .90 of answered trials in run 1, then 1.00 in runs 2–6. Normal for a
  native listener.
- **Missed responses labelled `fb_wrong`** (no response row, no `fb_noresp`): 7 in run 1, 0–2 in
  later runs; nearly all of FLT15's "wrong" trials are misses. This affects only their
  `fb_wrong` regressor, so only `fbcorrectvswrong`, which natives aren't analysed on. `sound`
  and `fbcorrect` are valid (misses sit in their own regressor either way).
- Optional cleanup: reconvert FLT15's events so misses are `fb_noresp` (the converter needs a
  one-line change to treat a missing key as no response), then rerun their first level and
  extraction. Headline numbers won't change.
- Sensitivity, for the record: STGp `sound` group difference q=.036 → .079 without FLT15; FMC
  holds; the planned aCAU−aPUT difference would widen.

To do first: one `09_integration/subjects.csv` with group, and per-analysis inclusion flags
(behavior / striatal GLM / FLT2 PR / FST1), so every n in the paper traces to one file.

Sample gaps outside this repo (as of 2026-10-30):
- FLT2: FLT02 still absent from the outputs (stale-cache bug, being rerun); FLT22 rerun status
  unknown. §9a's merge with `participants.tsv` labels only 21 of 23 loaded subjects.
  `sub_list` in the notebook setup cell is stale (FLT06 loaded but not listed).
- FLT06/11/14: ran cleanly in FLT2; QA glance (tSNR, voxel counts, R² maps) still owed.
- FLT27 appears in some FLT2 lists; confirm its status.

Power, stated once and used everywhere: 12 vs 12, two-sided α=.05, 80% power needs d≈1.2 for a
group difference; interactions need more. So keep the number of primary group tests small,
report effect sizes with CIs, and support every **"absent in natives"** claim with an equivalence
test (TOST) or Bayes factor, not a non-significant p.

## 2. Analyses, section by section

*Out of scope: the HCP structural-connectivity (DWI) analyses move to a separate manuscript.
This paper uses only this study's own fMRI and behavioral data.*

### R1. Behavior
- Accuracy: `04_behavior/two_group_accuracy.py` (2026-11-13; run, results in §0b) reads the corrected
  `run_accuracies.tsv`, n=12/13 including FLT15, and writes per-run means with 95% CIs, run 6 vs
  run 1 within each group, the group × run mixed ANOVA (GG-corrected) and the two-group figure
  `04_behavior/behav_acc_two-group.{png,svg}`. Its accuracy is per-category accuracy averaged
  over categories, on answered trials; learners match the old .38→.82.
- RT: group × run model still to port from the old repo's `behavior_analysis.ipynb`.
- Response confusion matrices for both groups (old repo has the code and outputs).
- SPC / decision-bound modeling: **learners only** -- natives at ceiling make strategy fits
  uninformative. Say so explicitly. Optionally show natives' best-fitting model descriptively;
  the BIC file already includes them.

### R2. Striatal sound response (Tian S2) -- run 2026-11-04
- Done: all 26 participants through `univariate_glm.py --event_type=feedback` (one model,
  `scrubbed-splitfb`, grouped by stage), which saves `sound` alongside the feedback contrasts.
  The sound discrepancy (REVISION_PLAN #21) is resolved by this model.
- Results: see §0b. aPUT and NAc-core respond in both groups; no striatal ROI differs between
  groups in overall level or at any single stage; no striatal group × stage interaction
  survives FDR.
- Prediction check: "sound response in posterior striatum in both groups" is only partly met --
  pPUT is significant in natives only, and pCAU in neither once tests use one value per
  participant (REVISION_PLAN #33).
- TOST implemented 2026-11-12 (two-group section 5d, `gs.equivalence_tests`); not yet run on the
  real data. **Equivalence bounds -- fixed here before running; don't change them after seeing
  the results.** Raw beta units, each tied to a learner effect and computed in the notebook:

  | Claim | Test (striatal regions; FDR within claim) | Bound |
  |---|---|---|
  | Striatal sound level similar in both groups | learners − natives, stage-averaged `sound` (Welch) | half the striatal mean sound response (all six regions, both groups) |
  | Natives' striatal feedback response doesn't change | natives' `fbcorrect` `[-1,0,1]` trend vs 0 | \|learners' headline aCAU `fbcorrectvswrong` trend\| (≈0.59) |
  | Natives' striatal sound response doesn't change | natives' `sound` trend vs 0 | \|learners' pPUT `sound` trend\| (≈1.41, their largest sound change) |
  | pPUT sound decline similar in both groups | learners − natives, `sound` trend | same as above |

  Report each result as the verdict plus `bound_reached` (the smallest bound that equivalence
  would pass) and BF01. With 12 vs 13 expect equivalence only at large bounds (≈ d 1.2
  two-sample, 0.85 one-sample); an inconclusive TOST is worded "we can't rule out a difference
  smaller than `bound_reached`".
- **TOST results (2026-11-12):**
  - Sound level: no region equivalent at ±0.66. `bound_reached` 1.26 (aPUT) to 2.87 (pCAU); BF01
    1.4–2.6. "Present in both groups" can only mean "significant within each group"; don't
    write "similar".
  - Natives' feedback change: **equivalent in aCAU** (trend 0.07, 90% CI [−0.26, 0.40], q=.024,
    BF01=3.4) and NAc-core (q=.024, BF01=3.6); the other four just miss (q=.084, `bound_reached`
    0.64–0.67 vs 0.59). Natives' aCAU `fbcorrect` change is smaller than the learners' headline
    effect. **Caveat:** the bound comes from learners' `fbcorrectvswrong` (0.59), but learners'
    own `fbcorrect` aCAU trend is only −0.33, inside natives' `bound_reached` (0.40). So this
    shows natives lack a change of the headline's size, not that they differ from learners on
    the same contrast (that's the planned test, p=.13). Don't re-pick the bound after the fact.
  - Natives' sound change: none equivalent at ±1.41 (`bound_reached` 1.38–2.30; BF01 2.0–3.4).
  - pPUT sound decline, learners − natives: inconclusive (`bound_reached` 2.08, BF01=2.1).
    The pPUT result is neither learner-specific nor shown to be shared.
- Done 2026-11-12 (two-group 4b): pPUT sound trend × group. Learners decline (t=−3.52), natives
  don't significantly (t=−1.03), but the groups don't differ (t=−0.88, p=.39). Report as a
  within-learner effect, not a learner-specific one; secondary (chosen after the §0c results).
- Done 2026-11-12 (two-group 5c, `gs.laterality_tests`): group × hemisphere. Striatal sound:
  natives left > right (t=2.62), learners not (t=0.65), group p=.22. No striatal laterality claim
  beyond "natives' response is left-weighted".

### R3. Striatal feedback -- needs a new contrast
**Constraint:** natives get ~1 wrong-feedback trial per run (~2 per stage), so
`fb_correct − fb_wrong` cannot be estimated for them by stage.
- **Contrast A (both groups): `fb_correct` vs baseline**, per stage. Learning account: correct
  feedback is informative early for learners and becomes predictable → aCAU decline in learners;
  uninformative throughout for natives → low and flat.
- **Contrast B (learners only): `fb_correct − fb_wrong`** -- the existing headline, unchanged.
- First check that A, in learners, reproduces B's aCAU pattern. If it doesn't, the cross-group
  story rests on A and the Discussion must say what A measures.
- ~~Code~~ Done: `fbcorrect` is saved by `univariate_glm.py`. Contrasts naming a regressor missing
  from a run are skipped and logged (`_estimable_contrasts`). `fb_wrong` coverage in natives:
  `fbcorrectvswrong` exists for 6/13 early, 3/13 middle, 1/13 final; six natives have none.
- ~~Stats~~ Done in `group_stats.trend_group_difference` (Welch t, group-label permutation 10k,
  bootstrap CI, LOSO), with two-group functions added to `resampling_stats.py`, instead of a
  `--group_col` option on `robustness_checks.py`.

**Results (2026-11-04):**
- **The A-reproduces-B check failed.** In learners, aCAU `fbcorrect` trend −0.33 (t=−1.91,
  p=.083) vs `fbcorrectvswrong` −0.59 (t=−3.30, p=.007); the two trends correlate r=.24 (p=.45).
- Because contrasts are linear, B − A = −`fb_wrong`, so learners' implied wrong-feedback trend is
  about +0.26. **The headline decline is roughly half less response to correct feedback and half
  more response to wrong feedback.** Natives have too few errors to measure the second half.
- Planned group tests on A: aCAU difference not significant (t=−1.59, p=.13, LOSO 1/25).
  aCAU−aPUT differs robustly (−0.66, t=−2.52, p=.019, permutation p=.018, LOSO 25/25,
  CI [−1.15, −0.17]): learners −0.39 (p=.052), natives +0.27 (p=.18).
- Components (2026-11-17): learners aCAU −0.33, aPUT +0.07; natives aCAU +0.07, aPUT −0.20.
  None is significant alone, so the result is the *relative* change. In learners it's mostly aCAU
  falling; in natives it's mostly aPUT falling while aCAU stays flat (TOST-equivalent to 0, R2).
- Convergence (descriptive): learners − natives aCAU `fbcorrect` by stage is +0.30, +0.15, −0.11
  (t=1.40, 0.62, −0.41). Learners' aCAU starts above natives' and ends at their level. No stage
  differs significantly; a formal test would be the trend difference above (aCAU alone p=.13).
  Fine to show in a figure and describe; don't claim it as a result.

**Framing options (decide before writing; §6):**
1. Lead with the aCAU−aPUT differential `fbcorrect` trend as the cross-group result, and keep B
   as the learner-only headline. Say plainly that A captures only the correct-feedback part.
2. Add `fbwrong` (wrong feedback vs baseline, learners only) as a derived contrast and report
   the decomposition of B into its correct and wrong parts. That makes clear which part natives
   can and can't speak to.
3. Treat the two-group feedback result as supporting rather than decisive: robust for the
   differential trend, absent for aCAU alone.
Options 1 and 2 combine naturally. **Decided 2026-11-17: 1 + 2** (see Framing at the top). To do:
learners' `fbwrong` aCAU trend (implied ≈ +0.26) tested with the same battery (sign-flip, LOSO,
bootstrap) in the two-group notebook.

### R4. Cortical and PFC ROI networks -- run 2026-11-04; per-group 2026-11-11
Between-group tests are in `group_level_two_group.ipynb` (IFGo/IFGt dropped as duplicates of
ParsOp/ParsTri). Since 2026-11-11, `group_level_all_ROI.ipynb` also runs the full per-group
analysis (ANOVAs, post hocs, Table 5/8-style tables, brain plots) for either group via
`analysis_group`; within-group results are in §0c.
- PFC sound: learners' FMC/OFC/FP deactivation is mostly absent in natives (§0c). Worth framing
  as a learner-specific frontal response to the sounds, with FMC as the FDR-surviving group
  difference.
- Per-ROI group differences (section 5b; stage-averaged and per stage): `sound` FMC and STGp
  survive FDR (§0b); OFC `sound` is close (q=.060). Nothing survives per stage.
- Group × stage ANOVA: only FMC `sound` (group main effect) survives FDR.
- **Naming:** FMC is presumably Harvard-Oxford "Frontal Medial Cortex" (the other PFC labels
  match that atlas; confirm against the mask source). Anatomically it's ventromedial PFC. In the
  text and on slides, write "ventromedial prefrontal cortex (vmPFC; Harvard-Oxford FMC)" once,
  then vmPFC. Task-related vmPFC deactivation is common, which gives the learner-only
  deactivation an interpretation (effortful listening) -- untested, so Discussion only.
- Learners' `fbcorrect` is below baseline in auditory cortex (HG t=−6.15, PP, PT); natives'
  isn't significant but has the same sign.

### R5. Auditory representational dimensionality (FLT2)
- Replace the per-stage 2-way decompositions (§9g) with one model:
  `PR ~ group * stage * level + (1 | subject)` (level = Primary/Association; ROI-level as a
  follow-up), statsmodels MixedLM or R lme4.
- Key question: **is the early→final PR decline learner-specific?** If natives decline as much,
  it's repetition/adaptation; if only learners, it's learning-related compression.
- Existing signal to confirm on the full sample: non-Mandarin higher PR in right auditory
  cortex (R-STGa p_FDR=.017), a Primary-level group difference at the middle stage
  (p_FDR=.019), and Primary > Association in learners vs the reverse in natives.
- Current FLT2 result (n=23, groups pooled, 25.2.5/T1w, notebook re-run 2026-09-15): stage
  main effect F(2,44)=5.37, p=.008 (GG p=.009); stage × ROI n.s. (p_GG=.29). PR is lower in final
  than early in every region; per ROI early-vs-final FDR across 14 ROIs: R-HG, R-STGa, R-ParsOp
  (p_FDR=.048), L-STGp, R-ParsTri (.050). Primary composite early-vs-final t(22)=2.39, p=.026,
  d=.35. These pool groups; the per-group picture is the open question above.
- Fixes owed before any number goes in the paper (FLT2 `representational_dimensionality.ipynb`):
  - Sample: FLT02/FLT22, group labels, `sub_list` (§1).
  - §13 D1: the "all-runs" row repeats the final-stage values (np²=.195 instead of §9c's .229);
    variable reuse bug.
  - Stale markdown: §8e-iii says no ROI survives FDR (five do); §13 cites a §9c p=.003 that no
    longer matches.
  - Not executed on the recompute: §9h (D_sheng cross-validation), §9i (PR ↔ accuracy), §11
    searchlight, §13 D2. Searchlight D_sheng loads 0 subjects (searchlight not re-run; group maps
    would need MNI).
- Report PR as primary and ROI-level D_sheng (already in `dimensionality_roi.py` CSVs) as
  robustness; skip the searchlight version.
- **Where the code lives (2026-11-17):** per-subject processing stays in FLT2; group-level
  analysis is `10_multivariate/dimensionality_two_group.ipynb` here (planned Primary-composite
  trend test, the mixed model above, exploratory per-region tests, D_sheng, PR trend export for
  R7), via `multivariate_data.py` and `group_stats.py`. Tested on synthetic data only.
- **Before RSA or decoding (optional section below):** FLT2 masks only `--atlas=dseg`, so there are
  no striatal (Tian-S2) or IC/MGN trial betas yet; FLT2's `rsa_roi.py` still reads the old
  untagged `glmsingle/` tree (22.1.1/MNI) and builds one RDM per session, not per stage; decoding
  exists only in `auditory_category_learning/multivariate/` (nilearn LSS, MNI). The old
  `07_rsa/` is archived (`archive/07_rsa/`).

### R6. Acoustic baseline (FST1, stgrid)
Stgrid has no learning and no tone categories, so it answers: **are the FLT2 group differences
already present for basic spectrotemporal sound structure (trait/baseline), or specific to tone
categories (experience)?**
- PR: group × ROI on stgrid; then within-subject tonecat-vs-stgrid × group.
- **Trial matching is mandatory:** stgrid has 2 runs (2 presentations per stimulus); tonecat has
  ~18 per stimulus. Recompute tonecat PR from 2 presentations per stimulus (e.g. first
  presentation of each stimulus in runs 1–2) before any cross-task comparison.
- Tuning maps (`analyze_stgrid_spectrotemporal.py`): group × ROI on spectral/temporal preference
  and selectivity. Port to the 25.2.5/T1w GLMsingle dir, and replace the current per-ROI test
  against the grand mean of all ROIs (non-independent reference) with an RM/mixed ANOVA or a
  stimulus-label permutation.
- All current FST1 outputs are 22.1.1/MNI; nothing is reportable until rerun.
- Status (2026-10-30): stgrid GLMsingle (T1w, block design via `stimdur`; one block per stimulus
  per run, so already single-trial) re-run; masking + `dimensionality_roi.py` not recorded; RD
  notebook never executed; `analyze_stgrid_spectrotemporal.py` still reads `glmsingle_stgrid/` in
  MNI.
- **Integrity checks (open):** fMRIPrep ran twice into the same `denoised_fmriprep-25.2.5/` tree
  (`--task tonecat` in this repo, then FST1's `run_fmriprep_stgrid.sh` with `--task stgrid`).
  1. Did the stgrid pass rewrite anatomical outputs this repo depends on (`desc-preproc_T1w`,
     `from-MNI152NLin2009cAsym_to-T1w` xfm, brain mask)? Compare mtimes/checksums with the
     tonecat-era files. If they changed, the T1w masks from `05_masking/warp_atlases_to_T1w.sh`
     were built on a transform that no longer matches -- re-warp or show equivalence.
  2. tonecat vs stgrid `space-T1w` BOLD affine/shape, one file each per subject (nibabel, five
     minutes). There's one mask directory per subject, resampled to whichever task's file sorts
     first; if the grids differ, the fix is per-task mask trees, not a different reference.

### R7. Integration
- **Striatum ↔ cortex (analysis B), learners.** Write this spec, dated, at the top of
  `integration_B.ipynb` before computing any correlation:
  - Striatal scalar: aCAU `fbcorrectvswrong` `[-1,0,1]` trend (main; decided 2026-11-17);
    aCAU−aPUT differential (secondary). From `roi_trend_scalars.csv`.
  - Cortical scalar: the same `[-1,0,1]` trend on FLT2 PR, Primary composite (HG+PT+PP, L/R
    pooled; main). R-PP alone as one pre-specified secondary test.
  - Predicted sign: **positive r** (PR and aCAU feedback both decline). Confirm PR still declines
    in learners alone (R5) before running B.
  - Specificity: the original "association composite should be null" check no longer works --
    association regions decline at least as much as primary (STGa d=.45, ParsOp d=.42 vs HG
    d=.34). Use (a) a non-auditory control ROI from `carpet_dseg` (needs its own GLMsingle/PR
    run), or (b) the striatal side: aPUT trend should *not* correlate with PR (costs nothing).
  - Test: Pearson (Spearman as robustness), bootstrap 95% CI, LOSO, exact sign-flip permutation
    (`resampling_stats.py`); table via `stats_fmt.py`. n=12, correlational: "co-varies with".
  - Natives go on the same scatter descriptively (little trend variance; slope difference
    underpowered).
- **Merge layer** (`09_integration/`): `subjects.csv` (§1); `roi_crosswalk.csv` (Tian S2 ↔
  auditory `dseg` ↔ carpet labels); this repo's trend export `roi_trend_scalars.csv` -- done
  2026-11-03 (last cell of `group_level_all_ROI.ipynb`); FLT2's `sub-*_dimensionality_PR.csv` read
  directly; `integration_B.ipynb` merges on `participant_id` and runs the battery.
- **Brain ↔ behavior, learners:** accuracy slope (run 1→6) vs aCAU trend and vs PR trend.
- **Group difference ↔ baseline:** does the FLT2 group difference in PR survive covarying the
  same subject's stgrid PR (R6)?
- Later, only if time allows: SPC strategy (`late_minus_early` Δ-BIC, learners) ↔ aCAU feedback
  trend (REVISION_PLAN Workstream 1, R2's suggested link) and ↔ PR trend.

### Optional: RSA (decide early)
Native-vs-learner is the classic RSA comparison: a Tone-category model should fit natives from
run 1 and emerge across runs in learners. PR says how many dimensions there are, not whether
they are the tone categories. RSA was cut once for "drift", but here it would be the most direct
category test. If kept, port the outlier/NaN-masking fix from
`auditory_category_learning/rsa/rsa_roi.py` and recompute on GLMsingle 25.2.5/T1w betas.

## 3. Manuscript architecture

- **Intro:** striatal feedback learning + auditory cortical category representations;
  native-listener control as the design's answer to "is it learning?". Hypotheses stated per
  group (the table in §0), before any results.
- **Methods:** participants and language background; same task for both groups; GLMsingle + PR;
  FST1 task; stage split; feedback Contrast A vs B and why; equivalence testing. State that
  response onsets are modeled as a nuisance regressor and not reported (sound–response
  separation <1 s at TR=3 s), and that all ROI tests use one value per participant.
- **Methods additions for FLT2/FST1:** GLMsingle single-trial estimation (HRF library,
  GLMdenoise, fracridge; one column per trial) and how it differs from the nilearn
  condition-level GLM used for the striatum; PR definition and ceiling (min(n_stimuli,
  n_voxels); 16 stimuli in both tasks) and the voxel-count check (FLT2 §8d-ii); shared stage split
  (runs 1–2 / 3–4 / 5–6) and `[-1,0,1]` trend; same per-subject T1w `carpet_dseg` masks as the
  univariate analyses; stgrid task, block design, `stimdur`, separate fMRIPrep pass.
- **Results order:** R1 behavior → R2/R3 striatum → R4 cortical/PFC networks →
  R5 cortex dimensionality → R6 baseline → R7 integration.
- **Figures:** every main figure becomes two-group (learners and natives side by side, same
  axes). Fig: behavior; Fig: striatal sound/feedback by stage × group;
  Fig: PR by stage × group; Fig: stgrid baseline; Fig: integration scatter.
  Status (2026-11-17):
  - Behavior: `04_behavior/behav_acc_two-group.{png,svg}` -- done.
  - Striatal feedback by stage × group: `stage_profiles_by_group.png` (two-group section 6) has
    the right data but isn't paper-ready: four panels (keep the two `fbcorrect` panels, aCAU and
    aPUT), unshared y-axes, code-name titles ("aCAU · fbcorrect"), saved to the stats folder.
    Make a two-panel version with a shared y-axis and readable titles in `figures/two-group/`.
    Pair it with the aCAU−aPUT panel from `group-diff_key-results.png`.
  - Group differences: `group-diff_forest.png` (supplement), `group-diff_key-results.png` and the
    ROI maps in `06_univariate/figures/two-group/`. Surface maps are PNG only (SVGs exceed
    GitHub's limit) and need cropping. The `tian-S2 fbcorrect trend` map shows per-region
    differences, none significant, so it's supplementary, not a main panel.
  - PR by stage × group: box+strip per region (existing FLT2 style) plus group eigenspectra by
    stage. Not started.
  - Analysis B: scatter of aCAU trend vs PR trend (learners; natives descriptive), LOSO-stable fit
    and bootstrap band; supplement reuses `plot_robustness_checks.py`. Not started.
  - Stgrid baseline: not started.
- **Discussion:** what generalizes across groups (auditory input to striatum, cortical PR
  structure) vs what is learner-specific; limits of n=12 per group.

## 4. Work order

**Phase 0 -- setup (this week)**
1. Commit `09_integration/` and `REVISION_PLAN.md`.
2. Build `subjects.csv`; ~~FLT15~~ (kept, §1) / FLT27 / FLT02.
3. ~~Count `fb_wrong` per subject × stage, both groups.~~ Done via `roi-values_missing.tsv` (R3).
4. Integrity checks (R6: stgrid fMRIPrep anatomical overwrite; tonecat vs stgrid grid).

**Phase 1 -- decisive, cheap tests (go/no-go for the two-group framing)**
5. Behavior group × run (port existing).
6. FLT2 group × stage × level mixed model on existing outputs.
7. ~~Striatal GLMs for Mandarin subjects (sound + feedback with Contrast A); group × stage on aCAU.~~
   Done 2026-11-04 (R2, R3): aCAU alone n.s.; aCAU−aPUT differs (p=.019); A ≠ B in learners.
8. **Go/no-go.** *Striatal half answered (step 7): neither "learner-specific" nor "shared" cleanly --
   see R3 framing options. The PR half (step 6) is still open.* If the aCAU decline and the PR decline are learner-specific → two-group paper.
   If natives show the same declines → still a two-group paper, but the story becomes
   time-on-task/repetition, and the learning claims in the current manuscript need softening.
   Either outcome is informative; decide framing here, before writing.

**Phase 2 -- full analyses**
9. ~~R3 robustness with group~~ (done, `group_stats.py`). Equivalence tests implemented
   2026-11-12 (two-group 5d; bounds in R2); run on the cluster. Cortical claims (auditory ROIs,
   §0b) not yet covered.
9b. ~~Sound trend group test (pPUT) and group × hemisphere (§0c, R2).~~ Done 2026-11-12: both
    null as group differences.
9c. ~~Per-group executed notebooks.~~ Done: `group_level_all_ROI.ipynb` (learners) and
    `group-Mand_ROI.ipynb` (natives).
9d. Figures: paper-ready stage profiles (two panels, shared y-axis, readable titles; §3) and
    cropped surface maps. vmPFC naming throughout (R4).
10. ~~R4 networks with group.~~ Done 2026-11-04 (R4).
11. R6 FST1: masking + PR for all, trial-matched tonecat PR, tuning port and stats fix.
12. R7 integration.
13. RSA, if chosen.

**Phase 3 -- writing**
14. Rewrite Intro/hypotheses for the two-group design; Methods; Results; Discussion.
15. Response-to-reviewers mapping (Reviewer 1 replication/specificity → R3 group × stage;
    Reviewer 2 integration → R7). Note in the response that the connectivity analyses reviewers
    commented on now belong to a separate manuscript.
16. Merge FLT2 branch and `manuscript-revision` once numbers are frozen.

## 5. Risks

- **Feedback contrast change -- realized.** `fb_correct` doesn't reproduce the learners'
  `fb_correct − fb_wrong` aCAU decline (R3). Keep B as the learner-only result so the original
  finding stays intact, and frame the group result around what A measures.
- ~~**FLT15.**~~ Resolved 2026-11-13: responses fine, scoring bug fixed, kept (§1). The STGp
  sound difference still weakens without them (q=.079); say so if it's reported.
- **Response timing.** `resp` isn't reported, but RT differences between groups can still shift
  variance between the `sound` and `resp` regressors. Group differences in `sound` (FMC, STGp)
  carry a milder version of this caveat.
- **Unit of analysis.** Collapsing over stage, hemisphere or region must average within
  participant first (REVISION_PLAN #33). The old RSA "all regions" tests have the same problem.
- **Ceiling in natives** limits any correlation or trend analysis within that group; use them
  as a reference, not a second sample for correlations.
- **Confounds beyond learning:** natives also differ in RT (~0.9 vs ~1.2 s), task difficulty,
  and possibly arousal. RT can enter as a covariate for striatal analyses; difficulty cannot be
  separated from learning in this design -- say so.
- **Two beta estimators.** FLT2/FST1 use GLMsingle single-trial betas; the striatum uses the
  nilearn condition-level GLM. Fine for subject-level scalar correlations (each side is internally
  consistent), but say so in Methods and don't imply one unified model. Same fMRIPrep version,
  space (T1w) and masks on both sides, so no cross-space caveat.
- **Causality.** One cohort, correlational: "consistent with / co-varies with", never "drives".
- **Scope.** This is a bigger rewrite than the non-Mandarin plan. Its payoff is that it answers
  the reviewers' central doubt; it should replace, not add to, weaker sections.

## 6. Decisions for you
1. ~~Two-group framing vs the non-Mandarin-only plan.~~ Decided 2026-11-17: both groups, plus
   learners-only analyses (Framing, top).
2. ~~Feedback framing.~~ Decided 2026-11-17: R3 options 1 + 2, including the `fbwrong`
   decomposition.
2b. ~~FLT15~~ Decided 2026-11-13: included everywhere (§1).
2c. ~~`resp`~~ Decided 2026-11-04: modeled, not reported.
3. RSA in or out.
4. FST1 as a main-text baseline (recommended in this design) or supplement.
5. Specificity check for analysis B: control ROI (a) or aPUT-null (b) (R7).

## Appendix: repos and what each provides

Same cohort, same sessions, same `data_denoised/sub-*` list in all repos.

| Repo | Role | Per-subject outputs used |
|---|---|---|
| `fMRI_auditory-category-learning` (this repo, `manuscript-revision`) | Home of the manuscript: striatal/cortical GLMs, robustness checks, SPC model comparison, masks | ROI betas by stage (`roi-values_scrubbed-splitfb/`), trend scalars (`roi_trend_scalars.csv`), SPC `late_minus_early` Δ-BIC |
| `FLT2` (branch `representational-dimensionality`, unmerged) | R5: PR of tonecat responses by stage | `sub-*_dimensionality_PR.csv`, `_Dsheng.csv`, `_eigenspectra.npz` per `stage-*/`; GLMsingle 25.2.5/T1w in `glmsingle_fmriprep-25.2.5_space-T1w` |
| `FST1` | R6: stgrid acoustic baseline | stgrid PR, tuning maps (once rerun in T1w) |
| `auditory_category_learning` (`sitek/`, original pipeline) | Source material only | RSA outlier/NaN-masking fix (`rsa/rsa_roi.py`, needed only if RSA returns); combined-group behavior summary |

