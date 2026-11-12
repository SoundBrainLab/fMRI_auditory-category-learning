#!/usr/bin/env python
''' QA of the tonecat events files the first-level GLMs use: per participant
x run, trial and feedback counts and accuracy, plus one participant's
response confusion matrix (correct key x key pressed) per run.

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
parser.add_argument('--sub', default='FLT15', help='participant for the confusion matrices')
args = parser.parse_args()

rows = []
for fpath in sorted(glob(os.path.join(args.bids_dir, 'sub-*', 'func', 'sub-*_task-tonecat_run-*_events.tsv'))):
    events = pd.read_csv(fpath, sep='\t', index_col=0)
    fb = events.trial_type.value_counts()
    resp = events[events.trial_type.astype(str).str.startswith('resp_')]
    rows.append({'participant_id': os.path.basename(fpath).split('_')[0],
                 'run': os.path.basename(fpath).split('_run-')[1][:2],
                 'n_sound': events.trial_type.astype(str).str.startswith('sound_').sum(),
                 'n_resp': len(resp),
                 'resp_keys': ','.join(sorted(resp.trial_type.str[5:].unique())),
                 'correct_keys': ','.join(sorted(resp.correct_key.dropna().astype(int).astype(str).unique())),
                 **{k: fb.get(k, 0) for k in ['fb_correct', 'fb_wrong', 'fb_noresp', 'fb_none']}})
summary = pd.DataFrame(rows)
summary['accuracy'] = summary.fb_correct / (summary.fb_correct + summary.fb_wrong + summary.fb_noresp)

pd.set_option('display.width', 200, 'display.max_rows', 500)
print('runs with accuracy < .25 or unusual keys:')
usual_keys = summary.resp_keys.mode()[0]
print(summary[(summary.accuracy < .25) | (summary.resp_keys != usual_keys)].to_string(index=False))
print(f'\n(most common resp_keys: {usual_keys})')

print(f'\nsub-{args.sub}, all runs:')
print(summary[summary.participant_id == f'sub-{args.sub}'].to_string(index=False))

for fpath in sorted(glob(os.path.join(args.bids_dir, f'sub-{args.sub}', 'func',
                                      f'sub-{args.sub}_task-tonecat_run-*_events.tsv'))):
    events = pd.read_csv(fpath, sep='\t', index_col=0)
    resp = events[events.trial_type.astype(str).str.startswith('resp_')]
    print(f'\n{os.path.basename(fpath)}: correct key (rows) x key pressed (columns)')
    print(pd.crosstab(resp.correct_key, resp.trial_type.str[5:]))

out_fpath = os.path.join(args.bids_dir, 'derivatives', 'behavior', 'qa_events_summary.tsv')
summary.to_csv(out_fpath, sep='\t', index=False)
print('\nsaved', out_fpath)
