"""
Formatting helpers for consistent APA-style statistical reporting.

Usage:
    from stats_fmt import fmt_t, fmt_F, fmt_p, fmt_r, stat_str

    print(fmt_t(11, 2.345))           # "t(11) = 2.35"
    print(fmt_F(2, 22, 4.123))        # "F(2, 22) = 4.12"
    print(fmt_p(0.0003))              # "p < .001"
    print(fmt_r(0.7823))              # "r = .78"
    print(stat_str('t', 11, 2.345, 0.038))  # "t(11) = 2.35, p = .038"
"""


def fmt_p(p: float) -> str:
    """Format a p-value to 3 decimal places, APA style (no leading zero).
    Values below .001 reported as 'p < .001'."""
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def fmt_t(df: int | float, t: float) -> str:
    """Format a t-statistic with degrees of freedom."""
    return f"t({df}) = {t:.2f}"


def fmt_F(df1: int | float, df2: int | float, F: float) -> str:
    """Format an F-statistic with numerator and denominator degrees of freedom."""
    return f"F({df1}, {df2}) = {F:.2f}"


def fmt_r(r: float) -> str:
    """Format a correlation coefficient to 2 decimal places, no leading zero."""
    return f"r = {r:.2f}".replace("r = 0.", "r = .").replace("r = -0.", "r = -.")


def fmt_z(z: float) -> str:
    """Format a z-statistic to 2 decimal places."""
    return f"z = {z:.2f}"


def stat_str(stat_type: str, *args) -> str:
    """
    Convenience wrapper that returns a full 'stat, p' string.

    Examples
    --------
    stat_str('t', df, t_val, p_val)
    stat_str('F', df1, df2, F_val, p_val)
    stat_str('r', r_val, p_val)
    """
    if stat_type == 't':
        df, t_val, p_val = args
        return f"{fmt_t(df, t_val)}, {fmt_p(p_val)}"
    elif stat_type == 'F':
        df1, df2, F_val, p_val = args
        return f"{fmt_F(df1, df2, F_val)}, {fmt_p(p_val)}"
    elif stat_type == 'r':
        r_val, p_val = args
        return f"{fmt_r(r_val)}, {fmt_p(p_val)}"
    elif stat_type == 'z':
        z_val, p_val = args
        return f"{fmt_z(z_val)}, {fmt_p(p_val)}"
    else:
        raise ValueError(f"Unknown stat_type '{stat_type}'. Use 't', 'F', 'r', or 'z'.")


def fmt_pingouin_anova(aov_df, term_col: str = 'Source') -> dict[str, str]:
    """
    Convert a pingouin ANOVA DataFrame to a dict of formatted strings keyed by term name.
    Each value is ready to paste into manuscript text.

    Parameters
    ----------
    aov_df : pd.DataFrame
        Output of pg.anova(), pg.rm_anova(), or pg.mixed_anova()
    term_col : str
        Column name that contains the effect labels (default 'Source')

    Returns
    -------
    dict mapping effect name -> formatted string, e.g.
        {'learning_stage': 'F(2, 22) = 4.12, p = .038'}
    """
    out = {}
    for _, row in aov_df.iterrows():
        name = row[term_col]
        # pingouin uses 'ddof1'/'ddof2' or 'DF' depending on the test
        df1 = int(row.get('ddof1', row.get('DF', '?')))
        df2_key = 'ddof2' if 'ddof2' in row else ('DF2' if 'DF2' in row else None)
        df2 = int(row[df2_key]) if df2_key else '?'
        F = row.get('F', float('nan'))
        p = row.get('p-unc', row.get('p-GG-corr', float('nan')))
        out[name] = stat_str('F', df1, df2, F, p)
    return out
