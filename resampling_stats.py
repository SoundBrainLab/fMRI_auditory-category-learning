import itertools

import numpy as np
from scipy import stats

''' Generic resampling/permutation checks on a per-subject scalar,
tested against zero. Shared by 06_univariate/robustness_checks.py
(ROI learning-stage trends) and 08_behavioral-modeling/model_comparison.py
(SPC relative-BIC trend) -- factored out here rather than duplicated,
following this repo's existing convention of root-level shared utilities
(see stats_fmt.py). '''


def leave_one_out_stability(values, alpha=0.05):
    ''' refit a one-sample t-test against 0, leaving out each subject in
    turn. Reports how many of the N folds preserve the original effect's
    direction and significance -- a stable effect across all folds is a
    concrete answer to "would this replicate"; one that flips when a
    particular subject drops out is exactly the fragility reviewers
    doubted. '''
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError(f'leave_one_out_stability: need at least 2 subjects, got {n}')
    orig_t, orig_p = stats.ttest_1samp(values, 0)
    orig_sign = np.sign(orig_t)

    n_preserved = 0
    for i in range(n):
        loo_t, loo_p = stats.ttest_1samp(np.delete(values, i), 0)
        if np.sign(loo_t) == orig_sign and loo_p < alpha:
            n_preserved += 1

    return {'original_t': orig_t, 'original_p': orig_p,
            'n_folds': n, 'n_preserved': n_preserved}


def exact_sign_flip_permutation_test(values):
    ''' exact two-sided permutation test for a one-sample scalar against
    0, enumerating all 2^n possible sign flips. Feasible up to about
    n=20 subjects (2^20 = ~1M); at n=12 (2^12 = 4096) this is exact
    rather than a Monte Carlo approximation -- a real upside of small n,
    not just a limitation. '''
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError(f'exact_sign_flip_permutation_test: need at least 2 subjects, got {n}')
    if n > 20:
        raise ValueError('exact enumeration infeasible above ~20 subjects; use Monte Carlo instead')

    observed = np.mean(values)
    n_perms = 0
    n_as_extreme = 0
    for signs in itertools.product([1, -1], repeat=n):
        permuted_mean = np.mean(np.asarray(signs) * values)
        if abs(permuted_mean) >= abs(observed) - 1e-12:
            n_as_extreme += 1
        n_perms += 1

    return {'observed_mean': observed, 'n_permutations': n_perms,
            'p_value': n_as_extreme / n_perms}


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    ''' percentile bootstrap CI on the mean of a per-subject scalar --
    defends an effect size without leaning on parametric normality
    assumptions at n=12. '''
    rng = np.random.default_rng() if rng is None else rng
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError(f'bootstrap_ci: need at least 2 subjects, got {len(values)}')
    boot_means = np.array([np.mean(rng.choice(values, size=len(values), replace=True))
                           for _ in range(n_boot)])
    alpha = 1 - ci
    return {'mean': float(np.mean(values)),
            'ci_lower': float(np.percentile(boot_means, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
            'n_boot': n_boot}


''' Two-group versions of the checks above: is a per-subject scalar
different between two independent groups (e.g. learners vs. native
listeners)? Each takes the two groups' values as separate arrays and
tests mean(a) - mean(b). '''


def _check_two_groups(name, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f'{name}: need at least 2 subjects per group, got {len(a)} and {len(b)}')
    return a, b


def leave_one_out_two_groups(a, b, alpha=0.05):
    ''' Welch t-test of a vs. b, refit leaving out each subject (from
    either group) in turn; counts folds preserving the original
    direction and significance. '''
    a, b = _check_two_groups('leave_one_out_two_groups', a, b)
    orig_t, orig_p = stats.ttest_ind(a, b, equal_var=False)
    orig_sign = np.sign(orig_t)

    n_preserved = 0
    folds = [(np.delete(a, i), b) for i in range(len(a))] + [(a, np.delete(b, i)) for i in range(len(b))]
    for fold_a, fold_b in folds:
        loo_t, loo_p = stats.ttest_ind(fold_a, fold_b, equal_var=False)
        if np.sign(loo_t) == orig_sign and loo_p < alpha:
            n_preserved += 1

    return {'original_t': orig_t, 'original_p': orig_p,
            'n_folds': len(folds), 'n_preserved': n_preserved}


def group_label_permutation_test(a, b, n_perms=10000, rng=None):
    ''' two-sided Monte Carlo permutation test on mean(a) - mean(b),
    shuffling group labels. Exact enumeration (24 choose 12 = ~2.7M
    splits at 12 per group) is possible but slow in pure Python, so this
    samples; p uses the (k + 1) / (n + 1) convention so it can never be
    reported as exactly 0. '''
    rng = np.random.default_rng() if rng is None else rng
    a, b = _check_two_groups('group_label_permutation_test', a, b)
    pooled = np.concatenate([a, b])
    observed = np.mean(a) - np.mean(b)

    n_as_extreme = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(pooled)
        permuted = np.mean(shuffled[:len(a)]) - np.mean(shuffled[len(a):])
        if abs(permuted) >= abs(observed) - 1e-12:
            n_as_extreme += 1

    return {'observed_difference': observed, 'n_permutations': n_perms,
            'p_value': (n_as_extreme + 1) / (n_perms + 1)}


def bootstrap_ci_two_groups(a, b, n_boot=10000, ci=0.95, rng=None):
    ''' percentile bootstrap CI on mean(a) - mean(b), resampling subjects
    within each group separately so group sizes stay fixed. '''
    rng = np.random.default_rng() if rng is None else rng
    a, b = _check_two_groups('bootstrap_ci_two_groups', a, b)
    boot_diffs = np.array([np.mean(rng.choice(a, size=len(a), replace=True))
                           - np.mean(rng.choice(b, size=len(b), replace=True))
                           for _ in range(n_boot)])
    alpha = 1 - ci
    return {'difference': float(np.mean(a) - np.mean(b)),
            'ci_lower': float(np.percentile(boot_diffs, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(boot_diffs, 100 * (1 - alpha / 2))),
            'n_boot': n_boot}
