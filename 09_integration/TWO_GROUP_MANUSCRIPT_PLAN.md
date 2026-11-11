# Two-group FLT manuscript plan: learners (non-Mandarin) vs native listeners (Mandarin)

*Drafted 2026-10-30; updated 2026-11-04 with the first two-group striatal and cortical ROI results
(REVISION_PLAN.md #28–#33). Alternative to `MANUSCRIPT_INTEGRATION_PLAN.md` (non-Mandarin only).
Reuses its repo map, the FLT2/FST1 status notes and the integrity checks (§1b–1e there), which
apply unchanged.*

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

## 0b. Where the evidence stands (updated 2026-11-04)

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
| Only learners learn | Accuracy run 1→6: learners .38→.82 (t=6.67, p<.001); natives .95→.98 (t=1.88, p=.087) | behavior, 12/12 | ● | Formal group × run model; FLT15 excluded from the behavior file for "bad responses" (§1) |
| Medial frontal cortex responds differently to sound | FMC sound: learners below baseline (−0.35), natives above (+0.40); d=−1.44, q=.013, holds without FLT15 (q=.025); also the only group main effect surviving FDR in the group × stage ANOVA (F=13.0, q=.033) | new, 12/13 | ● (strongest striatal/frontal ROI group difference) | Not predicted -- interpret cautiously; check it isn't driven by `resp` timing (RT differs by group) |
| Cortical PR topography differs | Group × ROI F(13,247)=5.63, p<.001. Learners higher PR in right auditory cortex (R-STGa p_FDR=.017, d=1.65; R-HG/R-STGp p_FDR=.055); natives higher in L-ParsOp (p_FDR=.055) | new, 11/10 | ● (strongest new-pipeline group result) | Complete sample (FLT02 etc.); FST1 baseline to say whether it's experience or trait |
| Primary vs association hierarchy reverses by group | Learners Primary > Association at middle/final (p=.04/.02 unc.); natives reverse at middle (p=.04 unc.); all p_FDR=.08. Primary-level group difference largest at middle stage (t=-3.38, p_FDR=.019) | new, 11/10 | ◐ | Single mixed model instead of per-stage tests |
| ~~Striatal response-period activity only in learners~~ | **Dropped (2026-11-04).** `resp` is in the model as a nuisance regressor but not reported: the response is <1 s after the sound at TR=3 s, and group RT differences would confound it. (The new pipeline did show a NAc-shell `resp` group effect, q=.008; not reported for these reasons.) | -- | -- | -- |
| aCAU feedback decline is learning-specific | Learners, headline `fbcorrectvswrong`: aCAU trend t=−3.30, exact p=.007, LOSO 12/12; aCAU−aPUT t=−2.90, p=.017, LOSO 12/12. **Cross-group test (`fbcorrect`):** aCAU trend learners −0.33 (p=.083), natives +0.07 (p=.70), difference t=−1.59, p=.13, LOSO 1/25 -- not significant. **aCAU−aPUT trend differs between groups:** −0.66, t=−2.52, p=.019, permutation p=.018, LOSO 25/25, 95% CI [−1.15, −0.17] | new, 12/13 | ◐ -- the differential trend is robust, but `fbcorrect` only partly captures the headline effect (R3) | Reframe around what `fbcorrect` measures (§2 R3); FLT15 sensitivity for the planned tests |
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
- FLT15 events check (§1), then TOST for the similarity claims.

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

## 1. Sample

| Group | Behavior | Imaging | Notes |
|---|---|---|---|
| non-Mandarin (learners) | 12: FLT02 04 06 09 11 12 13 14 20 25 28 30 | same 12 | FLT02 run-1 response coding corrupted (behavioral modeling n=11) |
| Mandarin (natives) | 12: FLT03 05 08 10 17 18 19 21 22 23 24 26 | 13 so far (+FLT15) | **FLT15:** see below |
| Excluded | FLT07 (bad QA) | | FLT27 appears in some FLT2 lists -- confirm status |

**FLT15 (found 2026-11-04).** The old repo's `behav_conversion/behavior_analysis.ipynb` excludes
FLT15 on purpose (`'sub-FLT15', # bad responses 07/17/26`, commit `47ab055`); its behavioral file
had already needed a one-off reconversion in 2025 (commit `7639a4b`). This repo's
`04_behavior/behavior_analysis.ipynb` and every imaging script still include them. The events
files are built from the responses, so if the responses are bad, FLT15's `fbcorrect` and
`fbcorrectvswrong` maps are suspect too (`sound` is unaffected). **To do:** check FLT15's
per-run `resp_*`/`fb_*` counts on the cluster, then exclude them everywhere (12 vs 12) or nowhere.
Sensitivity so far: STGp `sound` group difference q=.036 → .079 without FLT15; FMC holds.

To do first: one `09_integration/subjects.csv` with group, and per-analysis inclusion flags
(behavior / striatal GLM / FLT2 PR / FST1), so every n in the paper traces to one file.

Power, stated once and used everywhere: 12 vs 12, two-sided α=.05, 80% power needs d≈1.2 for a
group difference; interactions need more. So keep the number of primary group tests small,
report effect sizes with CIs, and support every **"absent in natives"** claim with an equivalence
test (TOST) or Bayes factor, not a non-significant p.

## 2. Analyses, section by section

*Out of scope: the HCP structural-connectivity (DWI) analyses move to a separate manuscript.
This paper uses only this study's own fMRI and behavioral data.*

### R1. Behavior
- Accuracy and RT: group × run mixed ANOVA (already computed in the old repo's
  `behavior_analysis.ipynb`; port into `04_behavior/`).
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
- Still to do: TOST for the "present in both groups" claims; the `[-1,0,1]` sound trend × group
  test (only the ANOVA has been run for sound).

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

**Framing options (decide before writing; §6):**
1. Lead with the aCAU−aPUT differential `fbcorrect` trend as the cross-group result, and keep B
   as the learner-only headline. Say plainly that A captures only the correct-feedback part.
2. Add `fbwrong` (wrong feedback vs baseline, learners only) as a derived contrast and report
   the decomposition of B into its correct and wrong parts. That makes clear which part natives
   can and can't speak to.
3. Treat the two-group feedback result as supporting rather than decisive: robust for the
   differential trend, absent for aCAU alone.
Options 1 and 2 combine naturally.

### R4. Cortical and PFC ROI networks -- run 2026-11-04
Done in `group_level_two_group.ipynb` rather than by adding group to `group_level_all_ROI.ipynb`
(that notebook stays learners-only for the existing tables). IFGo/IFGt are dropped as duplicates
of ParsOp/ParsTri.
- Per-ROI group differences (section 5b; stage-averaged and per stage): `sound` FMC and STGp
  survive FDR (§0b); OFC `sound` is close (q=.060). Nothing survives per stage.
- Group × stage ANOVA: only FMC `sound` (group main effect) survives FDR.
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
- Fixes owed first (from the other plan): FLT02 missing, group labels for all subjects, §13 D1
  bug, stale markdown, unexecuted §9h/§9i.

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

### R7. Integration
- **Striatum ↔ cortex (analysis B), learners:** aCAU stage trend vs auditory PR stage trend,
  pre-registered as in `MANUSCRIPT_INTEGRATION_PLAN.md` §3E. Natives have little trend variance,
  so show them on the same scatter descriptively rather than testing a slope difference
  (underpowered).
- **Brain ↔ behavior, learners:** accuracy slope (run 1→6) vs aCAU trend and vs PR trend.
- **Group difference ↔ baseline:** does the FLT2 group difference in PR survive covarying the
  same subject's stgrid PR (R6)?

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
- **Results order:** R1 behavior → R2/R3 striatum → R4 cortical/PFC networks →
  R5 cortex dimensionality → R6 baseline → R7 integration.
- **Figures:** every main figure becomes two-group (learners and natives side by side, same
  axes). Fig: behavior; Fig: striatal sound/feedback by stage × group;
  Fig: PR by stage × group; Fig: stgrid baseline; Fig: integration scatter.
- **Discussion:** what generalizes across groups (auditory input to striatum, cortical PR
  structure) vs what is learner-specific; limits of n=12 per group.

## 4. Work order

**Phase 0 -- setup (this week)**
1. Commit `09_integration/` and `REVISION_PLAN.md`.
2. Build `subjects.csv`; resolve FLT15 (events check, §1) / FLT27 / FLT02.
3. ~~Count `fb_wrong` per subject × stage, both groups.~~ Done via `roi-values_missing.tsv` (R3).
4. Integrity checks from the other plan §1d (stgrid fMRIPrep anatomical overwrite; tonecat vs
   stgrid grid).

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
9. ~~R3 robustness with group~~ (done, `group_stats.py`); equivalence tests still to do.
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
- **FLT15.** Possibly bad responses (§1). One sound group difference (STGp) depends on them.
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
- **Scope.** This is a bigger rewrite than the non-Mandarin plan. Its payoff is that it answers
  the reviewers' central doubt; it should replace, not add to, weaker sections.

## 6. Decisions for you
1. Two-group framing vs the non-Mandarin-only plan (Phase 1 results can decide).
2. ~~Feedback Contrast A for the cross-group test.~~ Done. **New decision:** how to frame the
   feedback result now that A ≠ B in learners (R3 framing options 1–3), and whether to add the
   learners-only `fbwrong` decomposition.
2b. FLT15: include or exclude, everywhere (§1).
2c. ~~`resp`~~ Decided 2026-11-04: modeled, not reported.
3. RSA in or out.
4. FST1 as a main-text baseline (recommended in this design) or supplement.
