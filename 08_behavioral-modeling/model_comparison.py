import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resampling_stats import leave_one_out_stability, exact_sign_flip_permutation_test, bootstrap_ci

'''
Formal comparison of the decision-bound/SPC/random-response behavioral
models (Workstream 1 of the manuscript revision plan) -- directly
answers Reviewer #1's critique that SPC's improving fit was never
statistically compared against the alternatives.

Input: the subject-run model-fitting results file (tab-separated,
columns subject/Block/BestFit/BestFitBIC/AllFits/Param1-9), e.g.
2023-06-08_DecisionBoundModelingOutput.txt. `subject` values look like
"FLT02_03" (participant FLT02, run 3 -- ASSUMES the numeric suffix is
chronological run order, matching the convention used everywhere else
in this pipeline; not independently verifiable without the original
fitting script). `AllFits` is a dict-like string with BIC for every
candidate model, formatted with underscores standing in for
spaces/quote-delimiters (e.g. "{'UDX_1':_89.47,_'SPC':_82.65}").

Candidate model set is fixed at UDX_1-5, UDY_1-6, CON_1-5, SPC, RAN --
CJH1_1-4/CJH2_1-4 are excluded per user decision (erratic behavior:
disproportionate win rates and a repeated sentinel-like BIC value
traced to this family during initial review of a data subset).

Restricted to the non-Mandarin participant group (the manuscript's
analyzed sample, confirmed n=12), read dynamically from participants.tsv
rather than hardcoded, matching how the rest of this pipeline
(group_level_all_ROI.ipynb) determines group membership -- also means
this script doesn't need updating if the analyzed sample ever changes.

BIC sanity filtering: real fits in the reviewed subset ranged ~15-210;
non-convergence produced sentinel values far outside that (~2009-2016
repeated near-identically across unrelated subjects, or one wild
outlier at -6490 with malformed parameters). Any BIC outside
[--bic_min, --bic_max] is treated as a failed fit for that specific
model in that specific run -- excluded from comparison, not imputed.
Defaults (-50, 500) are based on that reviewed subset; sanity-check
against the full file before trusting them blindly.

Example (illustrative only -- fill in your actual paths, nothing here
is hardcoded as a default):
    python model_comparison.py \
        --fits_file=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/behavior/2023-06-08_DecisionBoundModelingOutput.txt \
        --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
        --out_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/behavior/
'''

CANDIDATE_MODELS = (
    ['UDX_%d' % i for i in range(1, 6)] +
    ['UDY_%d' % i for i in range(1, 7)] +
    ['CON_%d' % i for i in range(1, 6)] +
    ['SPC', 'RAN']
)
STAGE_MAP = {1: 'early', 2: 'early', 3: 'middle', 4: 'middle', 5: 'final', 6: 'final'}


def parse_allfits(allfits_str):
    ''' parse "{'UDX_1':_89.47,_'SPC':_82.65,...}" into {model: bic} '''
    pairs = re.findall(r"'([A-Za-z0-9_]+)':_([+-]?[\d.eE+-]+)", allfits_str)
    return {name: float(val) for name, val in pairs}


def load_non_mandarin_subjects(bidsroot):
    ''' subject IDs (without the 'sub-' prefix, matching the fits
    file's format) for the non-Mandarin group, read from
    participants.tsv -- same source of truth
    group_level_all_ROI.ipynb uses, not a hardcoded list. '''
    participants_fpath = Path(bidsroot) / 'participants.tsv'
    participants_df = pd.read_csv(participants_fpath, sep='\t')
    nman = participants_df.participant_id[participants_df.group == 'non-Mandarin']
    return sorted(s.replace('sub-', '') for s in nman)


def load_model_fits(fits_fpath, models=CANDIDATE_MODELS, bic_min=-50, bic_max=500):
    ''' long-format dataframe: subject, run, model, bic, valid.
    valid=False rows have a BIC outside [bic_min, bic_max] -- treated
    as a failed/non-converged fit, not a real comparison point.

    Run number comes from the numeric suffix of `subject` (e.g.
    "FLT02_03" -> run 3), NOT the `Block` column -- every row seen so
    far has Block == 'block1' identically, so it can't encode run
    number and is ignored. If the real file ever has more than one
    distinct Block value, that assumption may be wrong; this prints a
    warning rather than silently trusting it either way.

    Malformed rows (subject with no '_run' suffix, unparseable/missing
    AllFits) are skipped with a warning rather than crashing the whole
    load -- real data from this fitting procedure is already known to
    have at least one badly-behaved row (see module docstring). '''
    df = pd.read_csv(fits_fpath, sep='\t')

    if 'Block' in df.columns and df['Block'].nunique() > 1:
        print(f"WARNING: 'Block' column has {df['Block'].nunique()} distinct values "
              f"({sorted(df['Block'].unique())}) -- this script ignores it and infers "
              f"run number from the subject column suffix instead. Verify that's still "
              f"correct before trusting the results.")

    rows = []
    n_skipped = 0
    for _, row in df.iterrows():
        try:
            subject, run = row['subject'].rsplit('_', 1)
            run = int(run)
            allfits = parse_allfits(row['AllFits'])
        except (ValueError, AttributeError, TypeError) as e:
            print(f"WARNING: skipping malformed row (subject={row.get('subject')!r}): {e}")
            n_skipped += 1
            continue

        for model in models:
            if model not in allfits:
                continue
            bic = allfits[model]
            valid = bic_min <= bic <= bic_max
            rows.append({'subject': subject, 'run': run, 'model': model,
                        'bic': bic, 'valid': valid})

    if n_skipped:
        print(f"WARNING: skipped {n_skipped} malformed row(s) out of {len(df)} total")

    return pd.DataFrame(rows)


def restrict_to_subjects(long_df, subjects):
    return long_df[long_df['subject'].isin(subjects)].reset_index(drop=True)


def aggregate_model_comparison(long_df):
    ''' per subject x model, mean BIC across valid runs only (a model
    with zero valid runs for a subject is excluded from that subject's
    comparison rather than penalized for missing data) -- the direct
    "was there ever a real comparison" table Reviewer #1 asked for.
    Lower mean BIC wins. '''
    valid_df = long_df[long_df['valid']]
    agg = (valid_df.groupby(['subject', 'model'])
                    .agg(mean_bic=('bic', 'mean'), n_valid_runs=('bic', 'size'))
                    .reset_index())

    winners = (agg.loc[agg.groupby('subject')['mean_bic'].idxmin()]
                   [['subject', 'model', 'mean_bic']]
                   .rename(columns={'model': 'winning_model'})
                   .reset_index(drop=True))
    return agg, winners


def compute_spc_relative_bic(long_df):
    ''' per subject-run, delta = BIC_SPC - min(BIC of other candidate
    models), using only valid fits. Negative = SPC fits relatively
    better than the best alternative that run. Rows where SPC or every
    alternative is invalid are dropped, not imputed. Runs outside
    STAGE_MAP's expected {1..6} (e.g. a stray practice/aborted run left
    in the file) are skipped with a warning rather than crashing. '''
    rows = []
    n_unmapped = 0
    for (subject, run), g in long_df.groupby(['subject', 'run']):
        stage = STAGE_MAP.get(run)
        if stage is None:
            n_unmapped += 1
            continue
        spc_row = g[(g['model'] == 'SPC') & g['valid']]
        others = g[(g['model'] != 'SPC') & g['valid']]
        if spc_row.empty or others.empty:
            continue
        spc_bic = spc_row['bic'].iloc[0]
        best_other_bic = others['bic'].min()
        rows.append({'subject': subject, 'run': run, 'stage': stage,
                    'spc_bic': spc_bic, 'best_other_bic': best_other_bic,
                    'delta_bic': spc_bic - best_other_bic})
    if n_unmapped:
        print(f"WARNING: skipped {n_unmapped} subject-run(s) with a run number "
              f"outside STAGE_MAP's expected {{1..6}} -- check for stray/aborted runs")
    return pd.DataFrame(rows)


def compute_early_late_trend(delta_df):
    ''' per-subject mean delta_bic in the early vs. final stage, and the
    final-minus-early change -- more negative change means SPC's
    relative fit improved with learning, which is the actual
    "increasing procedural reliance" claim (not raw win-count).
    Subjects missing either stage entirely are dropped. Raises if that
    leaves no subjects at all, rather than silently returning an empty
    result that downstream stats would turn into misleading NaNs. '''
    if delta_df.empty:
        raise ValueError('compute_early_late_trend: delta_df is empty -- no valid '
                         'SPC-vs-competitor comparisons were computed at all')
    stage_means = (delta_df.groupby(['subject', 'stage'])['delta_bic']
                            .mean().unstack('stage'))
    for stage in ('early', 'final'):
        if stage not in stage_means.columns:
            stage_means[stage] = np.nan
    stage_means = stage_means.dropna(subset=['early', 'final'])
    if stage_means.empty:
        raise ValueError('compute_early_late_trend: no subject has both an early- and '
                         'final-stage valid SPC comparison -- cannot compute a trend')
    stage_means['late_minus_early'] = stage_means['final'] - stage_means['early']
    return stage_means.reset_index()


def run_model_comparison(fits_fpath, bidsroot, out_dir, bic_min=-50, bic_max=500,
                         n_boot=10000, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_non_mandarin_subjects(bidsroot)
    print(f'{len(subjects)} non-Mandarin subjects: {subjects}')

    long_df = load_model_fits(fits_fpath, bic_min=bic_min, bic_max=bic_max)
    long_df = restrict_to_subjects(long_df, subjects)
    long_df.to_csv(out_dir / 'model_fits_long.csv', index=False)
    n_invalid = int((~long_df['valid']).sum())
    print(f'{n_invalid} of {len(long_df)} model-fit rows flagged invalid '
          f'(BIC outside [{bic_min}, {bic_max}])')

    agg, winners = aggregate_model_comparison(long_df)
    agg.to_csv(out_dir / 'model_comparison_aggregate.csv', index=False)
    winners.to_csv(out_dir / 'model_comparison_winners.csv', index=False)
    print('winning-model counts (mean BIC across valid runs, per subject):')
    print(winners['winning_model'].value_counts())

    delta_df = compute_spc_relative_bic(long_df)
    delta_df.to_csv(out_dir / 'spc_relative_bic_by_run.csv', index=False)

    trend_df = compute_early_late_trend(delta_df)
    trend_df.to_csv(out_dir / 'spc_relative_bic_early_late.csv', index=False)

    values = trend_df['late_minus_early'].values
    rng = np.random.default_rng(seed)
    loso = leave_one_out_stability(values)
    exact_perm = exact_sign_flip_permutation_test(values)
    boot = bootstrap_ci(values, n_boot=n_boot, rng=rng)

    summary = pd.DataFrame([{
        'n_subjects': len(values),
        'mean_late_minus_early_delta_bic': float(np.mean(values)),
        'original_t': loso['original_t'],
        'original_p': loso['original_p'],
        'loso_folds_preserved': loso['n_preserved'],
        'loso_n_folds': loso['n_folds'],
        'exact_permutation_p': exact_perm['p_value'],
        'bootstrap_ci_lower': boot['ci_lower'],
        'bootstrap_ci_upper': boot['ci_upper'],
    }])
    summary.to_csv(out_dir / 'spc_trend_test_summary.csv', index=False)
    print('SPC relative-fit trend test (late vs. early stage, negative = SPC improved):')
    print(summary.to_string(index=False))

    return long_df, agg, winners, delta_df, trend_df, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Formal comparison of behavioral models (UDX/UDY/CON/SPC/RAN) '
                    'against Reviewer #1\'s critique, restricted to the non-Mandarin sample',
        epilog=('Example: python model_comparison.py '
                '--fits_file=/path/to/2023-06-08_DecisionBoundModelingOutput.txt '
                '--bidsroot=/path/to/data_denoised/ '
                '--out_dir=/path/to/data_denoised/derivatives/behavior/'))
    parser.add_argument('--fits_file', required=True,
                        help='path to the subject-run model-fitting results file')
    parser.add_argument('--bidsroot', required=True,
                        help='top-level BIDS directory containing participants.tsv')
    parser.add_argument('--out_dir', required=True,
                        help='directory to write output CSVs to')
    parser.add_argument('--bic_min', type=float, default=-50,
                        help='BIC values below this are treated as failed fits')
    parser.add_argument('--bic_max', type=float, default=500,
                        help='BIC values above this are treated as failed fits')
    parser.add_argument('--n_boot', type=int, default=10000, help='bootstrap resamples')
    parser.add_argument('--seed', type=int, default=0, help='random seed for bootstrap')
    args = parser.parse_args()

    run_model_comparison(args.fits_file, args.bidsroot, args.out_dir,
                         bic_min=args.bic_min, bic_max=args.bic_max,
                         n_boot=args.n_boot, seed=args.seed)
