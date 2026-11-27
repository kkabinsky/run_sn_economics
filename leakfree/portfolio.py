"""Portfolio level results: equal weight across the stocks of a cell.

The mean of the per-stock cumulative returns is not the return of a portfolio.
This module builds the daily net return of an equal-weight portfolio from the
stored forecasts, using the same trading rule as the corrected engine, and
reports its geometric return, its Sharpe ratio and a stationary-bootstrap
confidence interval.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import backtest, sharpe_of, stationary_bootstrap_ci


def daily_net_returns(actual, predicted, cost_bps: int = 20) -> np.ndarray:
    """The daily net return of one stock under the trading rule."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 3:
        return np.array([])
    prev = actual[:-1]
    edge = (predicted[1:] - prev) / np.abs(prev)
    market = actual[1:] / prev - 1.0
    pos = np.sign(edge)
    gross = pos * market
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return gross - turnover * cost_bps / 1e4


def portfolio_from_predictions(folder: Path, cost_bps: int = 20) -> pd.DataFrame:
    """One row per regime, window and model: the equal-weight portfolio."""
    rows = []
    directory = Path(folder) / "data" / "predictions"
    if not directory.is_dir():
        return pd.DataFrame()
    frames: dict = {}
    for path in sorted(directory.glob("*.csv")):
        stem = path.stem
        parts = stem.split("_")
        ticker = parts[0]
        regime = parts[1]
        window = int(parts[2].lstrip("W"))
        frame = pd.read_csv(path)
        for model in frame.columns:
            if model in ("date", "actual"):
                continue
            key = (regime, window, model)
            frames.setdefault(key, []).append(
                (ticker, frame["actual"].to_numpy(float),
                 frame[model].to_numpy(float)))

    for (regime, window, model), series in sorted(frames.items()):
        legs = [daily_net_returns(a, p, cost_bps) for _, a, p in series if len(a) > 2]
        if not legs:
            continue
        n = min(len(leg) for leg in legs)
        if n < 5:
            continue
        daily = np.mean([leg[:n] for leg in legs], axis=0)
        equity = np.cumprod(1.0 + daily)
        peak = np.maximum.accumulate(equity)
        low, high = stationary_bootstrap_ci(
            daily, lambda v: np.prod(1 + v) - 1, n_boot=2000, mean_block=10, seed=7)
        rows.append({
            "regime": regime,
            "window": window,
            "model": model,
            "stocks": len(legs),
            "n_days": int(n),
            "cost_bps": cost_bps,
            "portfolio_return": float(equity[-1] - 1.0),
            "ci_low": low,
            "ci_high": high,
            "sharpe": sharpe_of(daily),
            "max_drawdown": float((equity / peak - 1.0).min()),
            "mean_stock_return": float(np.mean(
                [backtest(a, p, cost_bps=cost_bps)["cum_return_geom"]
                 for _, a, p in series if len(a) > 2])),
        })
    return pd.DataFrame(rows)


def portfolio_frame_from_predictions(prediction_frames, cost_bps: int = 20) -> pd.DataFrame:
    """The same, for predictions already in memory."""
    rows = []
    grouped: dict = {}
    for (ticker, regime, window), frame in prediction_frames.items():
        for model in frame.columns:
            if model in ("date", "actual"):
                continue
            grouped.setdefault((regime, window, model), []).append(
                (ticker, frame["actual"].to_numpy(float),
                 frame[model].to_numpy(float)))
    for (regime, window, model), series in sorted(grouped.items()):
        legs = [daily_net_returns(a, p, cost_bps) for _, a, p in series if len(a) > 2]
        if not legs:
            continue
        n = min(len(leg) for leg in legs)
        if n < 5:
            continue
        daily = np.mean([leg[:n] for leg in legs], axis=0)
        equity = np.cumprod(1.0 + daily)
        peak = np.maximum.accumulate(equity)
        low, high = stationary_bootstrap_ci(
            daily, lambda v: np.prod(1 + v) - 1, n_boot=2000, mean_block=10, seed=7)
        rows.append({
            "regime": regime, "window": window, "model": model,
            "stocks": len(legs), "n_days": int(n), "cost_bps": cost_bps,
            "portfolio_return": float(equity[-1] - 1.0),
            "ci_low": low, "ci_high": high,
            "sharpe": sharpe_of(daily),
            "max_drawdown": float((equity / peak - 1.0).min()),
        })
    return pd.DataFrame(rows)
