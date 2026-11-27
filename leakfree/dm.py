"""Diebold-Mariano tests in the three conventions that appear in this project.

The reviewer asked (R3.M4) which loss function, which sign convention, which
small-sample correction and which reference distribution produced the
statistics in Table 2 of the manuscript.  Three different conventions are in
use on this machine, so all three are implemented here and named explicitly:

``dm_stat_paper``
    The convention of ``C:\\dmtest_python\\dmtest_summary_matrix.py``
    (``dm_z_stat``), which produced the published summary matrix
    ``C:\\program_stock\\dmtest_summary_matrix.xlsx``:
        d_t   = e1_t^2 - e2_t^2
        varD = population variance of d (ddof = 0), plus 2 * sum of the
                autocovariances for h > 1
        z     = mean(d) / sqrt(varD / T)        referenced to N(0, 1)
    No Bartlett weights, no small-sample correction.

``dm_stat_matlab``
    The convention of ``C:\\dmtest_python\\dmtest.m`` (Diebold and Mariano
    1995, as implemented by Semin Ibisevic 2011): the same loss differential
    and the same estimator, but ``var(d)`` uses the N-1 divisor and the
    autocovariances are divided by T.  The statistic is therefore
    sqrt((T-1)/T) times the paper's value.  Also referenced to N(0, 1).

``dm_stat_corrected``
    The Harvey-Leybourne-Newbold corrected statistic with a Newey-West
    Bartlett HAC variance and a t(T-1) reference distribution, from
    ``leakfree/evaluation.py``.  This is the version that should be reported
    in the revision.

Sign convention, identical in all three:
    d_t = loss_A - loss_B, so a positive statistic means the first model has
    the larger loss and is therefore the worse model.

The loss is the squared error in every case, matching the archived code.
"""
from __future__ import annotations

import numpy as np

from .evaluation import dm_test as dm_stat_corrected


def _aligned(e1, e2):
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    n = min(len(e1), len(e2))
    if n == 0:
        return np.array([]), np.array([])
    x, y = e1[:n], e2[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def dm_stat_paper(e1, e2, h: int = 1) -> float:
    """The published convention: population variance, N(0,1) reference."""
    x, y = _aligned(e1, e2)
    T = x.size
    if T < 3:
        return float("nan")
    d = x ** 2 - y ** 2
    d_mean = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    if h > 1:
        gammas = [float(np.dot(d[i:], d[:-i]) / T) for i in range(1, h)]
        var_d = gamma0 + 2.0 * float(np.sum(gammas))
    else:
        var_d = gamma0
    if var_d <= 0:
        return float("nan")
    return float(d_mean / np.sqrt((1.0 / T) * var_d))


def dm_stat_matlab(e1, e2, h: int = 1) -> float:
    """The MATLAB convention of dmtest.m: sample variance, N(0,1) reference."""
    x, y = _aligned(e1, e2)
    T = x.size
    if T < 3:
        return float("nan")
    d = x ** 2 - y ** 2
    d_mean = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))          # MATLAB var() uses N-1
    if h > 1:
        gammas = [float(np.dot(d[i:], d[:-i]) / T) for i in range(1, h)]
        var_d = gamma0 + 2.0 * float(np.sum(gammas))
    else:
        var_d = gamma0
    if var_d <= 0:
        return float("nan")
    return float(d_mean / np.sqrt((1.0 / T) * var_d))


def dm_p_value_normal(stat: float) -> float:
    """Two-sided p-value against the standard normal, as the archived code used."""
    from scipy import stats

    if not np.isfinite(stat):
        return float("nan")
    return float(2.0 * stats.norm.sf(abs(stat)))


def dm_statistic(e1, e2, h: int = 1) -> dict:
    """All three conventions for one pair of error series."""
    x, y = _aligned(e1, e2)
    paper = dm_stat_paper(x, y, h)
    matlab = dm_stat_matlab(x, y, h)
    corrected = dm_stat_corrected(x ** 2, y ** 2, horizon=h)
    d = x ** 2 - y ** 2
    return {
        "z_paper": paper,
        "p_paper_normal": dm_p_value_normal(paper),
        "z_matlab": matlab,
        "p_matlab_normal": dm_p_value_normal(matlab),
        "dm_hln": corrected["dm_hln"],
        "p_hln": corrected["p_value"],
        "mean_d": float(np.mean(d)) if d.size else float("nan"),
        "n": int(d.size),
        "better": "B" if np.isfinite(paper) and paper > 0 else
                  ("A" if np.isfinite(paper) and paper < 0 else "tie"),
    }


def paper_matrix_cell_errors(catboost_path, lightgbm_path, sheet: str = "BacktestData"):
    """The two error series behind one cell of the published matrix.

    The archived pipeline reads ``backtest_results_iter0.xlsx`` from the sheet
    ``BacktestData`` and prefers the column ``forecast_error``; when that
    column is missing it falls back to ``actual - predicted``.
    """
    import pandas as pd

    def load(path):
        try:
            frame = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            return None
        columns = {str(c).strip().lower(): c for c in frame.columns}
        if "forecast_error" in columns:
            return pd.to_numeric(frame[columns["forecast_error"]], errors="coerce").to_numpy()
        actual = columns.get("actual")
        predicted = columns.get("predicted") or columns.get("predict")
        if actual is not None and predicted is not None:
            a = pd.to_numeric(frame[actual], errors="coerce").to_numpy()
            p = pd.to_numeric(frame[predicted], errors="coerce").to_numpy()
            return a - p
        return None

    return load(catboost_path), load(lightgbm_path)
