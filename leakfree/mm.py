"""Sniper-style money management for the backtest.

A *sniper* trade is a small position placed precisely:

* the entry fires only when the signal is strong relative to the stock's own
  volatility (the forecast edge, measured in average true ranges, must pass a
  threshold) and when the market is not too quiet or too chaotic;
* the size follows from the risk the trade carries: a fixed fraction of the
  equity divided by the distance to the stop, so a wider stop means a smaller
  position;
* the stop is placed at a multiple of the average true range, the target at a
  multiple of it as well, and after one unit of profit the stop moves to the
  entry and then trails the price;
* the position is closed when the stop, the target or a time limit is reached.

On top of that the portfolio has its own limits: a maximum number of open
positions, a maximum weight per stock, a maximum gross exposure, a daily loss
limit and a cooling-off period after a run of losing trades.

The rules are few and explicit on purpose, because the reviewer asks how the
money is managed.  Nothing here looks at the test block before the moment it
acts: the average true range, the moving averages and the volatility filters
use the price history up to and including the day of the decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class SniperRules:
    """The money management rules, in one place."""

    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.01        # one percent of the equity per trade
    atr_window: int = 14                # trading days of the average true range
    stop_atr: float = 1.5               # the stop is 1.5 average true ranges away
    target_atr: float = 3.0             # the target is 3 average true ranges away
    trail_atr: float = 1.0              # the trailing stop, once the trade is in profit
    breakeven_at_r: float = 1.0         # the stop moves to the entry after one R
    max_holding_days: int = 10          # the time stop
    max_positions: int = 5              # open positions at the same time
    max_weight: float = 0.20            # of the equity, per stock
    max_gross: float = 0.80             # of the equity, in all positions
    daily_loss_limit: float = 0.02      # no new position after a loss of two percent
    losses_before_cooldown: int = 3     # consecutive losing trades
    cooldown_days: int = 5              # trading days without a new position
    edge_atr: float = 0.5               # the entry needs half an average true range
    min_atr_pct: float = 0.004          # the market may not be too quiet
    max_atr_pct: float = 0.08           # nor too chaotic
    trend_filter: bool = True           # the direction must agree with the trend
    trend_window: int = 20              # trading days of the moving average
    cost_bps: float = 20.0              # per unit of turnover

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    ticker: str
    side: int                 # +1 long, -1 short
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    target: float
    atr_at_entry: float
    risk_per_share: float
    entry_cost: float = 0.0
    best_price: float = 0.0
    breakeven: bool = False
    days: int = 0

    def market_value(self, price: float) -> float:
        """What the position is worth on the balance sheet."""
        return self.side * self.shares * price

    def pnl(self, price: float) -> float:
        """The profit of the trade if it is closed at this price."""
        return self.side * self.shares * (price - self.entry_price)

    def notional(self, price: float) -> float:
        return abs(self.shares * price)


def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       window: int = 14) -> np.ndarray:
    """The average true range, computed from the history up to each day."""
    n = len(close)
    true_range = np.empty(n)
    true_range[0] = high[0] - low[0]
    for index in range(1, n):
        previous = close[index - 1]
        true_range[index] = max(high[index] - low[index],
                                abs(high[index] - previous),
                                abs(low[index] - previous))
    result = np.empty(n)
    result[:] = np.nan
    if n < window:
        return result
    cumulative = np.cumsum(true_range)
    result[window - 1] = cumulative[window - 1] / window
    for index in range(window, n):
        result[index] = (result[index - 1] * (window - 1) + true_range[index]) / window
    return result


def moving_average(close: np.ndarray, window: int) -> np.ndarray:
    """The simple moving average, computed from the history up to each day."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cumulative = np.cumsum(close)
    result[window - 1] = cumulative[window - 1] / window
    for index in range(window, n):
        result[index] = (cumulative[index] - cumulative[index - window]) / window
    return result


def _mark(cash: float, positions: dict[str, Position], books: dict, date) -> tuple[float, float]:
    """The equity of the portfolio and the gross exposure, at the close."""
    equity = cash
    exposure = 0.0
    for ticker, position in positions.items():
        book = books.get(ticker)
        index = book["index"].get(date) if book else None
        price = book["close"][index] if index is not None else position.entry_price
        equity += position.market_value(price)
        exposure += position.notional(price)
    return equity, exposure


def run_sniper_mm(prices: dict, forecasts: dict[str, pd.Series],
                  rules: SniperRules | None = None) -> dict:
    """Run the money management over the forecast block.

    ``prices`` maps a ticker to a PriceSeries (with the open, high, low and
    close), ``forecasts`` maps a ticker to a series of forecast prices indexed
    by date.  The result holds the equity curve, the trade log and the
    summary measures.
    """
    rules = rules or SniperRules()
    if not forecasts:
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "metrics": {}}

    # the dates of the evaluation, the union over the stocks
    all_dates = sorted({pd.Timestamp(date)
                        for series in forecasts.values() for date in series.index})
    if not all_dates:
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "metrics": {}}

    # the signal of a day is the forecast made for the next trading day: the
    # decision is taken at the close of the day, on the forecast of the day
    # after, and the position is held from the next day onwards
    signal: dict[str, dict] = {}
    for ticker, series in forecasts.items():
        items = sorted((pd.Timestamp(date), float(value))
                       for date, value in series.items())
        signal[ticker] = {items[index][0]: items[index + 1][1]
                          for index in range(len(items) - 1)}

    # the price history of every stock, with the indicators
    books: dict[str, dict] = {}
    for ticker, series in prices.items():
        dates = pd.to_datetime(series.dates)
        close = np.asarray(series.close, dtype=float)
        high = np.asarray(series.high if series.high is not None else series.close,
                          dtype=float)
        low = np.asarray(series.low if series.low is not None else series.close,
                         dtype=float)
        position_of = {pd.Timestamp(date): index for index, date in enumerate(dates)}
        books[ticker] = {
            "dates": dates, "close": close, "high": high, "low": low,
            "index": position_of,
            "atr": average_true_range(high, low, close, rules.atr_window),
            "ma": moving_average(close, rules.trend_window),
        }

    cash = rules.initial_capital
    open_positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    consecutive_losses = 0
    cooldown_left = 0
    previous_equity = rules.initial_capital

    for date in all_dates:
        # 1) the exits of the open positions
        for ticker in list(open_positions):
            position = open_positions[ticker]
            book = books.get(ticker)
            if book is None:
                continue
            index = book["index"].get(date)
            if index is None:
                continue
            close = book["close"][index]
            high = book["high"][index]
            low = book["low"][index]
            position.days += 1

            exit_price = None
            reason = ""
            if position.side > 0:
                if low <= position.stop:
                    exit_price, reason = position.stop, "stop"
                elif high >= position.target:
                    exit_price, reason = position.target, "target"
            else:
                if high >= position.stop:
                    exit_price, reason = position.stop, "stop"
                elif low <= position.target:
                    exit_price, reason = position.target, "target"
            if exit_price is None and position.days >= rules.max_holding_days:
                exit_price, reason = close, "time"
            if exit_price is None:
                # the trailing stop follows the best price
                if position.side > 0:
                    position.best_price = max(position.best_price, high)
                else:
                    position.best_price = min(position.best_price, low) if position.best_price else low
                profit_atr = (position.best_price - position.entry_price) * position.side \
                    / max(position.atr_at_entry, 1e-9)
                if profit_atr >= rules.breakeven_at_r and not position.breakeven:
                    position.stop = position.entry_price
                    position.breakeven = True
                elif position.breakeven:
                    trail = position.best_price - position.side * rules.trail_atr * position.atr_at_entry
                    position.stop = (max(trail, position.entry_price) if position.side > 0
                                     else min(trail, position.entry_price))
                continue

            # the exit: the position is replaced by its market value in the cash
            gross = position.pnl(exit_price)
            cost = position.notional(exit_price) * rules.cost_bps / 1e4
            cash += position.market_value(exit_price) - cost
            net = gross - cost - position.entry_cost
            risk = position.risk_per_share * position.shares
            trades.append({
                "ticker": ticker, "side": position.side,
                "entry_date": position.entry_date, "exit_date": date,
                "entry_price": position.entry_price, "exit_price": exit_price,
                "shares": position.shares, "days": position.days, "reason": reason,
                "pnl": net, "gross_pnl": gross, "cost": cost + position.entry_cost,
                "r": (net / risk) if risk > 0 else np.nan,
            })
            consecutive_losses = consecutive_losses + 1 if gross < 0 else 0
            if consecutive_losses >= rules.losses_before_cooldown:
                cooldown_left = rules.cooldown_days
                consecutive_losses = 0
            del open_positions[ticker]

        # 2) the equity after the exits, which is what the daily loss limit sees
        equity_before, exposure_before = _mark(cash, open_positions, books, date)
        daily_return = (equity_before / previous_equity - 1) if previous_equity > 0 else 0.0
        previous_equity = equity_before

        # 3) the new positions
        blocked = (daily_return <= -rules.daily_loss_limit) or cooldown_left > 0
        if cooldown_left > 0:
            cooldown_left -= 1
        if not blocked and len(open_positions) < rules.max_positions:
            for ticker, series in sorted(forecasts.items()):
                if ticker in open_positions or ticker not in books:
                    continue
                forecast = signal.get(ticker, {}).get(date)
                if forecast is None:
                    continue
                book = books[ticker]
                index = book["index"].get(date)
                if index is None:
                    continue
                close = book["close"][index]
                atr_value = book["atr"][index]
                if not np.isfinite(atr_value) or atr_value <= 0:
                    continue
                atr_pct = atr_value / close
                if atr_pct < rules.min_atr_pct or atr_pct > rules.max_atr_pct:
                    continue
                edge = (forecast - close) / close
                if abs(edge) < rules.edge_atr * atr_value / close:
                    continue
                side = 1 if edge > 0 else -1
                if rules.trend_filter:
                    average = book["ma"][index]
                    if not np.isfinite(average):
                        continue
                    if side > 0 and close <= average:
                        continue
                    if side < 0 and close >= average:
                        continue
                stop_distance = rules.stop_atr * atr_value
                risk_budget = rules.initial_capital * rules.risk_per_trade
                shares = risk_budget / stop_distance
                notional = shares * close
                if notional > rules.max_weight * equity_before:
                    shares = rules.max_weight * equity_before / close
                    notional = shares * close
                if exposure_before + notional > rules.max_gross * equity_before:
                    shares = max(0.0, (rules.max_gross * equity_before - exposure_before) / close)
                    notional = shares * close
                if shares <= 0:
                    continue
                cost = notional * rules.cost_bps / 1e4
                # a long moves the notional out of the cash, a short moves it in
                cash -= side * shares * close + cost
                exposure_before += notional
                open_positions[ticker] = Position(
                    ticker=ticker, side=side, entry_date=date, entry_price=close,
                    shares=shares,
                    stop=close - side * stop_distance,
                    target=close + side * rules.target_atr * atr_value,
                    atr_at_entry=atr_value, risk_per_share=stop_distance,
                    entry_cost=cost, best_price=close,
                )

        # 4) the equity of the day, with the new positions marked at the close
        equity, gross_exposure = _mark(cash, open_positions, books, date)
        equity_rows.append({
            "date": date, "equity": equity,
            "positions": len(open_positions),
            "gross_exposure": gross_exposure / equity if equity > 0 else 0.0,
        })

    # the positions that are still open at the end of the block
    final_date = all_dates[-1]
    for ticker, position in list(open_positions.items()):
        book = books.get(ticker)
        index = book["index"].get(final_date) if book else None
        price = book["close"][index] if index is not None else position.entry_price
        gross = position.pnl(price)
        cost = position.notional(price) * rules.cost_bps / 1e4
        cash += position.market_value(price) - cost
        net = gross - cost - position.entry_cost
        risk = position.risk_per_share * position.shares
        trades.append({
            "ticker": ticker, "side": position.side, "entry_date": position.entry_date,
            "exit_date": final_date, "entry_price": position.entry_price,
            "exit_price": price, "shares": position.shares, "days": position.days,
            "reason": "end of block", "pnl": net, "gross_pnl": gross,
            "cost": cost + position.entry_cost,
            "r": (net / risk) if risk > 0 else np.nan,
        })

    equity_frame = pd.DataFrame(equity_rows)
    if not equity_frame.empty:
        equity_frame.loc[len(equity_frame) - 1, "equity"] = cash
    trade_frame = pd.DataFrame(trades)
    metrics = summarise_mm(equity_frame, trade_frame, rules)
    # the books have to balance: the final equity is the initial capital plus
    # the profit of every closed trade
    if len(trade_frame):
        expected = rules.initial_capital + float(trade_frame["pnl"].sum())
        metrics["accounting_difference"] = float(cash - expected)
        metrics["accounting_ok"] = bool(abs(cash - expected) < 1e-6 * max(abs(cash), 1.0))
    return {"trades": trade_frame, "equity": equity_frame, "metrics": metrics,
            "rules": rules.as_dict()}


def summarise_mm(equity: pd.DataFrame, trades: pd.DataFrame,
                 rules: SniperRules) -> dict:
    """The measures of the money management run."""
    if equity.empty:
        return {}
    values = equity["equity"].to_numpy(dtype=float)
    final = float(values[-1])
    total_return = final / rules.initial_capital - 1
    days = len(values)
    years = max(days / 252.0, 1e-9)
    cagr = (final / rules.initial_capital) ** (1.0 / years) - 1 if final > 0 else -1.0
    returns = np.diff(values) / values[:-1]
    sharpe = (returns.mean() / returns.std(ddof=1) * np.sqrt(252)
              if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0)
    peak = np.maximum.accumulate(values)
    max_drawdown = float((values / peak - 1).min())
    wins = int((trades["pnl"] > 0).sum()) if len(trades) else 0
    losses = int((trades["pnl"] < 0).sum()) if len(trades) else 0
    return {
        "final_equity": final,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades)) if len(trades) else np.nan,
        "average_r": float(trades["r"].mean()) if len(trades) and "r" in trades else np.nan,
        "average_holding_days": float(trades["days"].mean()) if len(trades) else np.nan,
        "time_in_market": float(equity["positions"].gt(0).mean()),
        "average_gross_exposure": float(equity["gross_exposure"].mean()),
    }
