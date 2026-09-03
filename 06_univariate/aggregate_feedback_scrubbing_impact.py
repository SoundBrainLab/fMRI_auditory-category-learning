import os
import re
import sys
import argparse
from glob import glob

import pandas as pd

'''
Combine per-subject qa_feedback_scrubbing_impact.py output across the
analyzed sample into one group-level view: is the FLT02 pattern (fb_wrong
trials thin out with learning + scrubbing severity rises over the session,
compounding worst in the middle/final stages) a general feature of this
dataset, or specific to that one subject?

Usage (point directly at a qc dir):
    python aggregate_feedback_scrubbing_impact.py \\
        --qc_dir=/PATH/TO/derivatives/nilearn/fmriprep-25.2.5/collapsed-nuisance/qc/ \\
        --task=tonecat --min_clean_trials=10

Usage (derive it from the same --fmriprep_dir/--variant_tag qa_feedback_scrubbing_impact.py used):
    python aggregate_feedback_scrubbing_impact.py \\
        --bidsroot=/PATH/TO/BIDS/DIR/ --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/ \\
        --variant_tag=collapsed-nuisance --task=tonecat
'''

FNAME_RE = re.compile(r'sub-(?P<sub>[^_]+)_task-(?P<task>[^_]+)_feedback-scrubbing-summary\.csv$')


def _fmriprep_tag(fmriprep_dir):
    ''' mirrors univariate_glm.py's _fmriprep_tag (duplicated, not imported,
    same reasoning as qa_feedback_scrubbing_impact.py). '''
    match = re.search(r'fmriprep-[\d.]+', os.path.basename(os.path.normpath(fmriprep_dir)))
    return match.group(0) if match else os.path.basename(os.path.normpath(fmriprep_dir))


def load_all_summaries(qc_dir, task):
    pattern = os.path.join(qc_dir, f'sub-*_task-{task}_feedback-scrubbing-summary.csv')
    fpaths = sorted(glob(pattern))
    if not fpaths:
        raise FileNotFoundError(
            f'no summary CSVs found matching {pattern} -- run '
            f'loop_qa_feedback_scrubbing_impact.sh first')

    frames = []
    for fpath in fpaths:
        m = FNAME_RE.search(os.path.basename(fpath))
        if not m:
            print(f'skipping unrecognized filename: {fpath}')
            continue
        df = pd.read_csv(fpath)
        df['participant_id'] = m.group('sub')
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate qa_feedback_scrubbing_impact.py output across subjects',
        epilog=('Example: python aggregate_feedback_scrubbing_impact.py '
                '--bidsroot=/PATH/TO/BIDS/DIR/ --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/ '
                '--variant_tag=scrubbed --task=tonecat'))
    parser.add_argument('--qc_dir',
                        help=('directory containing the per-subject summary CSVs. If omitted, '
                             'derived from --bidsroot/--fmriprep_dir/--variant_tag instead, '
                             'matching where qa_feedback_scrubbing_impact.py filed them'),
                        type=str, default=None)
    parser.add_argument('--bidsroot',
                        help='top-level BIDS dataset dir (only needed if --qc_dir is omitted)',
                        type=str, default=None)
    parser.add_argument('--fmriprep_dir',
                        help='fMRIprep derivatives dir (only needed if --qc_dir is omitted)',
                        type=str, default=None)
    parser.add_argument('--variant_tag',
                        help=("univariate_glm.py modeling-variant directory to read from -- "
                             "see that script's _variant_tag (only needed if --qc_dir is "
                             "omitted). Default matches the original baseline location"),
                        type=str, default='scrubbed')
    parser.add_argument('--task', help='task id', type=str, default='tonecat')
    parser.add_argument('--min_clean_trials',
                        help=('flag any subject/stage/condition with fewer than this many '
                             'trials surviving scrubbing (default 10)'),
                        type=int, default=10)
    parser.add_argument('--out_dir',
                        help='directory to write the combined CSV to (default: --qc_dir)',
                        type=str, default=None)
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    qc_dir = args.qc_dir or os.path.join(
        args.bidsroot, 'derivatives', 'nilearn', _fmriprep_tag(args.fmriprep_dir),
        args.variant_tag, 'qc')
    out_dir = args.out_dir or qc_dir
    os.makedirs(out_dir, exist_ok=True)

    combined = load_all_summaries(qc_dir, args.task)
    combined['n_clean'] = combined['n_trials'] - combined['n_affected']
    n_subjects = combined['participant_id'].nunique()

    combined_fpath = os.path.join(out_dir, f'all-subjects_task-{args.task}_feedback-scrubbing-summary.csv')
    combined.to_csv(combined_fpath, index=False)
    print(f'combined {n_subjects} subjects -> {combined_fpath}')

    stage_order = ['early', 'middle', 'final']
    combined['learning_stage'] = pd.Categorical(combined['learning_stage'],
                                                categories=stage_order, ordered=True)

    # group-level: pool trials across subjects (answers "overall, how much of this
    # condition survives scrubbing across the whole analyzed sample")
    group_level = (combined.groupby(['learning_stage', 'trial_type'], observed=True)
                   .agg(n_trials_total=('n_trials', 'sum'),
                        n_affected_total=('n_affected', 'sum'),
                        n_clean_total=('n_clean', 'sum'),
                        n_clean_min=('n_clean', 'min'),
                        n_clean_median=('n_clean', 'median'))
                   .reset_index())
    group_level['pct_affected_pooled'] = (100 * group_level['n_affected_total']
                                          / group_level['n_trials_total'])
    group_level = group_level.sort_values(['learning_stage', 'trial_type']).reset_index(drop=True)

    group_fpath = os.path.join(out_dir, f'group_task-{args.task}_feedback-scrubbing-summary.csv')
    group_level.to_csv(group_fpath, index=False)

    print(f'\ngroup-level summary across {n_subjects} subjects '
         f'(n_clean_min = worst single subject in that cell):')
    print(group_level.to_string(index=False))
    print(f'\nsaved {group_fpath}')

    # per-subject/stage flag: anyone whose fb_wrong (or fb_correct) clean trial
    # count drops below the threshold -- the subjects/stages most likely to have
    # an unstable beta estimate, worth a closer look or a robustness footnote
    fragile = combined[combined['n_clean'] < args.min_clean_trials].sort_values(
        ['learning_stage', 'trial_type', 'n_clean'])
    if len(fragile):
        print(f'\nsubject/stage/condition cells with fewer than {args.min_clean_trials} '
             f'clean trials after scrubbing:')
        print(fragile[['participant_id', 'learning_stage', 'trial_type',
                       'n_trials', 'n_affected', 'n_clean']].to_string(index=False))
        fragile_fpath = os.path.join(out_dir, f'flagged_task-{args.task}_feedback-scrubbing-fragile.csv')
        fragile.to_csv(fragile_fpath, index=False)
        print(f'saved {fragile_fpath}')
    else:
        print(f'\nno subject/stage/condition cells below {args.min_clean_trials} clean trials')


if __name__ == '__main__':
    main()
