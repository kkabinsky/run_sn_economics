"""Corrected backtest engine and Diebold-Mariano test for the SET50 cGAN/boosting study.

Replaces three defects in the legacy pipeline (D:/CatBoost/catBoost_conGAN.py):
  legacy :1181  signal = diff(predictions)      -> forecast edge pred[t] - actual[t-1]
  legacy :1198  threshold = 0.05*std(own preds) -> one common threshold for every model
  legacy :1208  returns clipped to +/-10%/day   -> no clipping
  legacy :1211  cumsum of returns               -> geometric compounding
  legacy :1219  Sharpe over non-zero days only  -> all days, flat days included
plus transaction costs, which the legacy engine never charged.

SIGN CONVENTION for dm_test: d_t = loss_A - loss_B, so DM > 0 means A has the
larger loss and B is the more accurate model. Reported explicitly in every table.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def backtest(actual, predicted, cost_bps=0.0, threshold_bps=0.0, horizon=1):
    """Long/short backtest driven by the forecast edge against the last observed price.

    predicted[t] is the forecast of actual[t] formed from information up to t-1,
    so the position for day t is decided at the close of t-1. No look-ahead.

    horizon > 1 holds the position for `horizon` days (non-overlapping entries).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)
    if n < 3:
        return None

    prev = actual[:-1]
    edge = (predicted[1:] - prev) / np.abs(prev)          # relative forecast edge
    market = actual[1:] / prev - 1.0                      # realised simple return

    pos = np.where(np.abs(edge) > threshold_bps / 1e4, np.sign(edge), 0.0)
    if horizon > 1:
        held = np.zeros_like(pos)
        for start in range(0, len(pos), horizon):
            held[start:start + horizon] = pos[start]
        pos = held

    gross = pos * market
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = turnover * cost_bps / 1e4
    net = gross - cost

    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    active = pos != 0

    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    err = predicted - actual
    naive_err = np.abs(np.diff(actual))                   # random-walk one-step error
    denom = naive_err.mean()

    return {
        "n_days": int(len(net)),
        "cum_return_geom": float(equity[-1] - 1.0),
        "cum_return_sum": float(net.sum()),               # legacy convention, for comparison
        "sharpe_annual": sharpe,
        "mean_daily": float(net.mean()),
        "vol_annual": float(sd * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "hit_rate": float((gross[active] > 0).mean()) if active.any() else 0.0,
        "exposure": float(active.mean()),
        "turnover_total": float(turnover.sum()),
        "n_trades": int((turnover > 0).sum()),
        "cost_drag": float(cost.sum()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "mase": float(np.abs(err).mean() / denom) if denom > 0 else np.nan,
        "equity": equity,
        "net_returns": net,
        "positions": pos,
    }


def buy_and_hold(actual):
    actual = np.asarray(actual, dtype=float)
    r = actual[1:] / actual[:-1] - 1.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    sd = r.std(ddof=1)
    return {
        "cum_return_geom": float(equity[-1] - 1.0),
        "sharpe_annual": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "max_drawdown": float((equity / peak - 1.0).min()),
        "equity": equity,
    }


def random_walk_forecast(actual):
    """Naive benchmark: the forecast of actual[t] is actual[t-1]."""
    actual = np.asarray(actual, dtype=float)
    return np.concatenate([[actual[0]], actual[:-1]])


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def newey_west_var(d, lag):
    d = np.asarray(d, dtype=float)
    t = len(d)
    dm = d - d.mean()
    gamma0 = (dm @ dm) / t
    total = gamma0
    for k in range(1, min(lag, t - 1) + 1):
        gk = (dm[k:] @ dm[:-k]) / t
        total += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(total, 1e-300)


def dm_test(loss_a, loss_b, horizon=1):
    """Harvey-Leybourne-Newbold corrected DM test.

    d_t = loss_a - loss_b.  DM > 0  =>  A is worse (larger loss)  =>  B is better.
    Uses a Newey-West HAC variance at lag = horizon - 1 and a t(T-1) reference,
    both of which the legacy tables omitted.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    d = loss_a - loss_b
    t = len(d)
    if t < 5 or not np.isfinite(d).all() or d.std() == 0:
        return {"dm": np.nan, "dm_hln": np.nan, "p_value": np.nan,
                "mean_d": float(np.nanmean(d)) if t else np.nan, "n": t, "better": "n/a"}

    var = newey_west_var(d, horizon - 1)
    stat = d.mean() / np.sqrt(var / t)
    correction = np.sqrt((t + 1 - 2 * horizon + horizon * (horizon - 1) / t) / t)
    stat_hln = stat * correction
    p = 2.0 * stats.t.sf(abs(stat_hln), df=t - 1)

    if p >= 0.05:
        better = "tie"
    else:
        better = "B" if stat_hln > 0 else "A"
    return {"dm": float(stat), "dm_hln": float(stat_hln), "p_value": float(p),
            "mean_d": float(d.mean()), "n": t, "better": better}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(running, 1.0)
    return out


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m / (rank + 1) * p[i])
        out[i] = running
    return out


# --------------------------------------------------------------------------- #
# stationary bootstrap (Politis & Romano 1994)
# --------------------------------------------------------------------------- #
def stationary_bootstrap_ci(x, statistic, n_boot=2000, mean_block=10, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        vals[b] = statistic(x[idx])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def sharpe_of(r):
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
