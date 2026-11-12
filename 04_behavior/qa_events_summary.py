#!/usr/bin/env python
''' QA of the tonecat events files the first-level GLMs use: per participant
x run, trial, response and feedback counts and accuracy, flagging runs that
look wrong; then, for one participant, the events rows with a missing
trial_type or onset and the response confusion matrix (correct key x key
pressed) per run.

A participant at ~0% accuracy who responds consistently shows a permuted
diagonal in the confusion matrix (a key-mapping mismatch); one who doesn't
shows a flat or single-column matrix (wrong keys, stuck button, no task).

Example: python qa_events_summary.py --sub FLT15
'''
import argparse
import os
from glob import glob

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
parser.add_argument('--bids_dir', default='/ix1/bchandrasekaran/krs228/data/FLT/data_denoised')
parser.add_argument('--sub', default='FLT15', help='participant for the detailed check')
args = parser.parse_args()


def key_label(trial_type):
    ''' 'resp_6' / 'resp_6.0' -> '6'; 'resp_None' / 'resp_nan' -> 'none' '''
    key = str(trial_type)[len('resp_'):].removesuffix('.0')
    return 'none' if key in ('None', 'nan', '') else key


def load_events(fpath):
    return pd.read_csv(fpath, sep='\t', index_col=0)


rows = []
for fpath in sorted(glob(os.path.join(args.bids_dir, 'sub-*', 'func', 'sub-*_task-tonecat_run-*_events.tsv'))):
    events = load_events(fpath)
    trial_type = events.trial_type.astype(str)
    resp = events[trial_type.str.startswith('resp_')]
    keys = resp.trial_type.map(key_label)
    correct_keys = set(resp.correct_key.dropna().astype(int).astype(str))
    fb = trial_type.value_counts()
    rows.append({'participant_id': os.path.basename(fpath).split('_')[0],
                 'run': os.path.basename(fpath).split('_run-')[1][:2],
                 'n_sound': trial_type.str.startswith('sound_').sum(),
                 'n_resp_rows': len(resp),
                 'n_no_resp': (keys == 'none').sum(),
                 'n_untyped': events.trial_type.isna().sum(),
                 'n_no_onset': events.onset.isna().sum(),
                 'float_keys': resp.trial_type.astype(str).str.endswith('.0').any(),
                 'other_keys': ','.join(sorted(set(keys) - correct_keys - {'none'})),
                 **{k: fb.get(k, 0) for k in ['fb_correct', 'fb_wrong', 'fb_noresp', 'fb_none']}})
summary = pd.DataFrame(rows)
summary['n_fb'] = summary[['fb_correct', 'fb_wrong', 'fb_noresp']].sum(axis=1)
summary['accuracy'] = summary.fb_correct / summary.n_fb

# a run is flagged if accuracy is below chance (4 keys), the events don't
# have one response and one feedback row per sound, rows are missing a
# trial_type, keys are stored as floats, or missed responses (key 'None')
# don't match the fb_noresp count
flags = {'below chance': summary.accuracy < .25,
         'resp rows != sounds': summary.n_resp_rows != summary.n_sound,
         'feedback rows != sounds': summary.n_fb != summary.n_sound,
         'untyped rows': summary.n_untyped > 0,
         'float keys': summary.float_keys,
         'no-resp != fb_noresp': summary.n_no_resp != summary.fb_noresp}
summary['problems'] = pd.DataFrame(flags).apply(lambda r: '; '.join(r.index[r]), axis=1)

pd.set_option('display.width', 250, 'display.max_rows', 500, 'display.max_columns', 30)
print('flagged runs:')
print(summary[summary.problems != ''].to_string(index=False))
print('\nkeys outside the correct-key set (stray presses; usually harmless):')
print(summary[summary.other_keys != ''][['participant_id', 'run', 'other_keys']].to_string(index=False))

print(f'\nsub-{args.sub}, all runs:')
print(summary[summary.participant_id == f'sub-{args.sub}'].to_string(index=False))

for fpath in sorted(glob(os.path.join(args.bids_dir, f'sub-{args.sub}', 'func',
                                      f'sub-{args.sub}_task-tonecat_run-*_events.tsv'))):
    events = load_events(fpath)
    print(f'\n=== {os.path.basename(fpath)}')
    odd = events[events.trial_type.isna() | events.onset.isna()]
    if len(odd):
        print('rows with no trial_type or no onset:')
        print(odd.to_string())
    resp = events[events.trial_type.astype(str).str.startswith('resp_')]
    print('correct key (rows) x key pressed (columns):')
    print(pd.crosstab(resp.correct_key, resp.trial_type.map(key_label)))

out_fpath = os.path.join(args.bids_dir, 'derivatives', 'behavior', 'qa_events_summary.tsv')
summary.to_csv(out_fpath, sep='\t', index=False)
print('\nsaved', out_fpath)
