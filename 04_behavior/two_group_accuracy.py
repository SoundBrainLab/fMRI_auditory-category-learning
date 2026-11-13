#!/usr/bin/env python
''' Tone-category accuracy by run for both groups, from the run_accuracies.tsv
that create_behav_confusion_matrices.py writes (one row per participant x
run; accuracy = mean of the confusion-matrix diagonal, i.e. per-category
accuracy averaged over categories, on trials with a response).

Writes, to derivatives/behavior/:
  accuracy_by_run_two-group.tsv      mean/SD/SEM/95% CI per group x run
  accuracy_tests_two-group.tsv       run 1 vs run 6 within each group (paired t),
                                     and the group x run mixed ANOVA
and the figure behav_acc_two-group.{png,svg} next to this script.

Run create_behav_confusion_matrices.py first (FLT15's keys are stored as
floats; scripts before commit f36d397 scored them all wrong).

Example: python two_group_accuracy.py
'''
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
parser.add_argument('--bids_dir', default='/ix1/bchandrasekaran/krs228/data/FLT/data_denoised')
args = parser.parse_args()

GROUPS = {'non-Mandarin': 'learners', 'Mandarin': 'natives'}
COLORS = {'non-Mandarin': '#2a6fdb', 'Mandarin': '#d9822b'}   # as in group_level_two_group.ipynb
beh_dir = os.path.join(args.bids_dir, 'derivatives', 'behavior')

acc = pd.read_csv(os.path.join(beh_dir, 'run_accuracies.tsv'), sep='\t')
participants = pd.read_csv(os.path.join(args.bids_dir, 'participants.tsv'), sep='\t')
acc = acc.merge(participants[['participant_id', 'group']], on='participant_id', how='left')
acc = acc[acc.group.isin(GROUPS)]
acc['run_num'] = acc.run.str.extract(r'(\d+)').astype(int)

# check the sample: every participant should have 6 runs
n_runs = acc.groupby('participant_id').run.nunique()
print(acc.groupby('group').participant_id.nunique().rename('n participants'))
if (n_runs != 6).any():
    print('WARNING: participants without 6 runs:', n_runs[n_runs != 6].to_dict())
print('\nFLT15 by run (should be near ceiling, not 0):')
print(acc[acc.participant_id == 'sub-FLT15'][['run', 'accuracy']].to_string(index=False))

# summary per group x run
def summarize(v):
    sem = v.std(ddof=1) / np.sqrt(len(v))
    half = stats.t.ppf(.975, len(v) - 1) * sem
    return pd.Series({'n': len(v), 'mean': v.mean(), 'sd': v.std(ddof=1), 'sem': sem,
                      'ci_lower': v.mean() - half, 'ci_upper': v.mean() + half})

summary = acc.groupby(['group', 'run_num']).accuracy.apply(summarize).unstack().reset_index()
summary.to_csv(os.path.join(beh_dir, 'accuracy_by_run_two-group.tsv'), sep='\t', index=False)
print('\n', summary.round(3).to_string(index=False))

# within each group: run 1 vs run 6 (paired); between: group x run mixed ANOVA
rows = []
for group in GROUPS:
    wide = acc[acc.group == group].pivot(index='participant_id', columns='run_num', values='accuracy')
    t, p = stats.ttest_rel(wide[6], wide[1])
    rows.append({'test': f'{GROUPS[group]}: run 6 vs run 1 (paired t)', 'n': len(wide),
                 'run1': wide[1].mean(), 'run6': wide[6].mean(), 'stat': t, 'df1': len(wide) - 1,
                 'df2': np.nan, 'p': p})
complete = acc[acc.participant_id.isin(n_runs[n_runs == 6].index)]
aov = pg.mixed_anova(data=complete, dv='accuracy', within='run_num', between='group',
                     subject='participant_id', correction=True)
# pingouin has renamed these across versions (DF1/ddof1, p-unc/p_unc, p-GG-corr/p_GG_corr)
col = lambda *names: next((c for c in names if c in aov.columns), None)
for _, r in aov.iterrows():
    p_gg = r[col('p_GG_corr', 'p-GG-corr')] if col('p_GG_corr', 'p-GG-corr') else np.nan
    rows.append({'test': f"group x run ANOVA: {r['Source']}", 'n': complete.participant_id.nunique(),
                 'stat': r['F'], 'df1': r[col('DF1', 'ddof1')], 'df2': r[col('DF2', 'ddof2')],
                 'p': r[col('p_unc', 'p-unc')], 'p_GG': p_gg, 'np2': r['np2']})
tests = pd.DataFrame(rows)
tests.to_csv(os.path.join(beh_dir, 'accuracy_tests_two-group.tsv'), sep='\t', index=False)
print('\n', tests.round(4).to_string(index=False))

# figure: per-participant lines (light) and group mean with 95% CI
fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=300)
for group, label in GROUPS.items():
    for _, sub in acc[acc.group == group].groupby('participant_id'):
        ax.plot(sub.run_num, sub.accuracy, color=COLORS[group], alpha=.18, lw=.8, zorder=1)
    s = summary[summary.group == group]
    ax.fill_between(s.run_num, s.ci_lower, s.ci_upper, color=COLORS[group], alpha=.18, lw=0, zorder=2)
    ax.plot(s.run_num, s['mean'], 'o-', color=COLORS[group], lw=2, ms=6, zorder=3)
    ax.annotate(label, (s.run_num.iloc[-1], s['mean'].iloc[-1]), xytext=(8, 0),
                textcoords='offset points', va='center', fontsize=9, color='0.15')
ax.axhline(.25, color='0.6', lw=.8, ls='--', zorder=0)
ax.annotate('chance', (1, .25), xytext=(0, -10), textcoords='offset points', fontsize=7, color='0.45')
ax.set_xticks(range(1, 7))
ax.set_xlim(.7, 7.0)
ax.set_ylim(0, 1.02)
ax.set_xlabel('run')
ax.set_ylabel('accuracy')
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
fig.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(out_dir, f'behav_acc_two-group.{ext}'), bbox_inches='tight')
print('\nsaved figure to', out_dir)
