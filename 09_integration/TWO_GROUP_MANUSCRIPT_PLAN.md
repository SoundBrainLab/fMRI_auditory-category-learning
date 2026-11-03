# Two-group FLT manuscript plan: learners (non-Mandarin) vs native listeners (Mandarin)

*Drafted 2026-10-30. Alternative to `MANUSCRIPT_INTEGRATION_PLAN.md` (non-Mandarin only). Reuses its
repo map, the FLT2/FST1 status notes and the integrity checks (§1b–1e there), which apply unchanged.*

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

## 0b. Where the evidence stands (2026-10-30)

Provenance: **new** = fMRIPrep 25.2.5, native T1w, current code (reportable once frozen).
**old** = fMRIPrep 22.1.1, MNI, `auditory_category_learning` notebooks (preliminary; numbers
will move on rerun). Strength: ● supports the claim · ◐ suggestive / indirect · ○ no evidence yet.

### Similarities between groups

| Claim | Current evidence | Prov. / n | Strength | Missing |
|---|---|---|---|---|
| Sound drives auditory cortex in both groups | Sound > baseline, every auditory ROI FDR-sig in both groups (natives t=3.3–12.7, all but ParsTri; learners t=2.7–7.6) | old, ~12/group | ● | Rerun on new pipeline; direct group test |
| Sound drives dorsal striatum in both groups | All caudate and putamen subdivisions FDR-sig in natives (t=2.8–7.3) and learners (t=4.0–9.5); NAc weaker in natives | old (LSS), ~12/group | ● | Rerun; equivalence test, since this is a "no difference" claim |
| Practice speeds responses in both groups | RT run 1→6: learners t=-3.29, p=.007; natives t=-3.05, p=.011 | behavior, 12/12 | ● | Group × run test. Implication: any neural effect that only tracks RT is not learning |
| Overall cortical dimensionality similar | FLT2 PR group main effect F(1,19)=0.85, p=.37 (but see topography below) | new, 11/10 | ◐ | Full sample; equivalence test |
| PR declines across stages | F(2,44)=5.37, p=.008, pooled across groups | new, 23 | ◐ for "shared" | **Group × stage** -- the decline could be shared (repetition) or learner-only (learning). Not yet tested |

### Differences between groups

| Claim | Current evidence | Prov. / n | Strength | Missing |
|---|---|---|---|---|
| Only learners learn | Accuracy run 1→6: learners .38→.82 (t=6.67, p<.001); natives .95→.98 (t=1.88, p=.087) | behavior, 12/12 | ● | Formal group × run model; FLT15 absent from behavior file |
| Cortical PR topography differs | Group × ROI F(13,247)=5.63, p<.001. Learners higher PR in right auditory cortex (R-STGa p_FDR=.017, d=1.65; R-HG/R-STGp p_FDR=.055); natives higher in L-ParsOp (p_FDR=.055) | new, 11/10 | ● (strongest new-pipeline group result) | Complete sample (FLT02 etc.); FST1 baseline to say whether it's experience or trait |
| Primary vs association hierarchy reverses by group | Learners Primary > Association at middle/final (p=.04/.02 unc.); natives reverse at middle (p=.04 unc.); all p_FDR=.08. Primary-level group difference largest at middle stage (t=-3.38, p_FDR=.019) | new, 11/10 | ◐ | Single mixed model instead of per-stage tests |
| Striatal response-period activity only in learners | `resp` regressor: learners anterior caudate/pCAU positive (t=3.4–3.9) and anterior putamen negative (t=-3.3 to -5.4), FDR-sig; natives nothing (all p_FDR≥.30) | old, ~12/group | ◐ | Rerun; direct group test; natives' easier decisions are a confound |
| aCAU feedback decline is learning-specific | Learners: aCAU decline t=-4.03, perm p=.0034, LOSO 12/12; aCAU−aPUT t=-4.21, p=.0024 | new, learners only | ○ for the *group* claim | **Natives never analyzed for feedback in either pipeline.** Needs Contrast A (§2 R3) |
| Tone-category structure emerges with learning | Learners: Tone RDM fits > Talker across cortex and striatum; Tone fit rises early→late | old, learners only | ◐ for learners, ○ for groups | Natives not analyzed. Old "all regions" tests pool ROI × subject rows as independent (e.g. t=14.6), which inflates them -- redo at subject level |
| Tone decoding differs by group | Natives' striatal tone decoding near chance (old, partial); cross-group "train native, test learner" decoding started with 5 natives / 7 learners | old, partial | ○ | Not reportable as is |
| Spectrotemporal (FST1) baseline differs | Pooled group sound map (n=21) only; group-difference section never updated; no stgrid PR yet | old | ○ | Everything |

### In progress
- FLT2 T1w recompute: FLT02/FLT22 reruns; notebook fixes (§13 D1 bug, stale text, unexecuted §9h/§9i).
- FST1: stgrid GLMsingle done in T1w; second fMRIPrep pass for stgrid (check anatomical overwrite); masking and PR not confirmed.
- Manuscript striatal pipeline: learners only so far; sound discrepancy (REVISION_PLAN #21) open.
- Behavioral model comparison: built and tested; confirm it has been run on the full file.

### Bottom line
- **Similarity claims are in decent shape:** sound responses in auditory cortex and striatum appear
  in both groups (old pipeline), and practice speeds both groups. They need reruns plus
  equivalence tests, not new ideas.
- **The strongest current group difference is cortical:** PR topography (new pipeline).
- **The paper's central claim has no native data yet.** Whether the aCAU feedback decline is
  learner-specific is untested. That single result (Phase 1, step 7) decides whether the
  two-group design strengthens or reframes the paper.

## 1. Sample

| Group | Behavior | Imaging | Notes |
|---|---|---|---|
| non-Mandarin (learners) | 12: FLT02 04 06 09 11 12 13 14 20 25 28 30 | same 12 | FLT02 run-1 response coding corrupted (behavioral modeling n=11) |
| Mandarin (natives) | 12: FLT03 05 08 10 17 18 19 21 22 23 24 26 | up to 13 | **FLT15** is Mandarin in `participants.tsv` and in FST1/FLT2 imaging, but absent from the behavior summary -- resolve |
| Excluded | FLT07 (bad QA) | | FLT27 appears in some FLT2 lists -- confirm status |

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

### R2. Striatal sound response (Tian S2)
- Run `univariate_glm.py --event_type=sound` (grouped by stage) for the Mandarin subjects;
  fMRIPrep 25.2.5 already exists for all 26, only the GLM loops are restricted to the 12.
- Test: group × stage × ROI; primary contrast on the per-subject `[-1,0,1]` stage-trend scalar.
- Prediction: sound response in caudal/posterior striatum in both groups (auditory input);
  stage-related change only in learners.
- Resolve the open sound discrepancy (REVISION_PLAN #21) first; it affects both groups equally.

### R3. Striatal feedback -- needs a new contrast
**Constraint:** natives get ~1 wrong-feedback trial per run (~2 per stage), so
`fb_correct − fb_wrong` cannot be estimated for them by stage.
- **Contrast A (both groups): `fb_correct` vs baseline**, per stage. Learning account: correct
  feedback is informative early for learners and becomes predictable → aCAU decline in learners;
  uninformative throughout for natives → low and flat.
- **Contrast B (learners only): `fb_correct − fb_wrong`** -- the existing headline, unchanged.
- First check that A, in learners, reproduces B's aCAU pattern. If it doesn't, the cross-group
  story rests on A and the Discussion must say what A measures.
- Code: add an `fbcorrect` contrast to the feedback branch of `univariate_glm.py`; handle runs with
  zero `fb_wrong` events (nilearn drops the column and a contrast string naming it fails).
  Count `fb_wrong` per subject × stage before running.
- Stats: group difference in the aCAU stage-trend scalar (Welch t + group-label permutation,
  10k Monte Carlo + bootstrap CI + LOSO), then the aCAU−aPUT differential trend × group.
  Extend `robustness_checks.py` with a `--group_col` option rather than writing a new script.

### R4. Cortical and PFC ROI networks
Same treatment as R2/R3 in `group_level_all_ROI.ipynb`'s auditory and PFC networks: add group as a
between factor; keep the network × contrast grid it already has.

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
  FST1 task; stage split; feedback Contrast A vs B and why; equivalence testing.
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
2. Build `subjects.csv`; resolve FLT15 / FLT27 / FLT02.
3. Count `fb_wrong` per subject × stage, both groups.
4. Integrity checks from the other plan §1d (stgrid fMRIPrep anatomical overwrite; tonecat vs
   stgrid grid).

**Phase 1 -- decisive, cheap tests (go/no-go for the two-group framing)**
5. Behavior group × run (port existing).
6. FLT2 group × stage × level mixed model on existing outputs.
7. Striatal GLMs for Mandarin subjects (sound + feedback with Contrast A); group × stage on aCAU.
8. **Go/no-go.** If the aCAU decline and the PR decline are learner-specific → two-group paper.
   If natives show the same declines → still a two-group paper, but the story becomes
   time-on-task/repetition, and the learning claims in the current manuscript need softening.
   Either outcome is informative; decide framing here, before writing.

**Phase 2 -- full analyses**
9. R3 robustness with group (`robustness_checks.py --group_col`), equivalence tests.
10. R4 networks with group.
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

- **Feedback contrast change.** Moving the cross-group test from `fb_correct − fb_wrong` to
  `fb_correct` changes what the headline measures. Keep B as the learner-only result so the
  original finding stays intact.
- **Ceiling in natives** limits any correlation or trend analysis within that group; use them
  as a reference, not a second sample for correlations.
- **Confounds beyond learning:** natives also differ in RT (~0.9 vs ~1.2 s), task difficulty,
  and possibly arousal. RT can enter as a covariate for striatal analyses; difficulty cannot be
  separated from learning in this design -- say so.
- **Scope.** This is a bigger rewrite than the non-Mandarin plan. Its payoff is that it answers
  the reviewers' central doubt; it should replace, not add to, weaker sections.

## 6. Decisions for you
1. Two-group framing vs the non-Mandarin-only plan (Phase 1 results can decide).
2. Feedback Contrast A for the cross-group test.
3. RSA in or out.
4. FST1 as a main-text baseline (recommended in this design) or supplement.
