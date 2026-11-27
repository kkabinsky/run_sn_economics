"""Chronological splits and a scaler that never sees the test period.

This module is the answer to R3.M2.  The archived pipeline

    legacy :153-160   scaler_X.fit_transform(X_windows_flat)     # every window
    legacy :165-170   train = all_indices[:905 + kk], test = next 100
    legacy :234-240   catboost_model.fit(..., eval_set=(X_test, y_test_orig))

made three mistakes at once:

  1. the MinMaxScaler was fitted on all sliding windows before the split, so
     the price range of the test period entered the training data;
  2. the test block was used as the early-stopping set, so the best iteration
     was chosen by looking at the test outcomes;
  3. there was no validation block at all.

Here the split is made on dates, the scaler is fitted on the training windows
only, and early stopping is driven by a validation block that sits strictly
between training and test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data import PriceSeries


@dataclass
class WindowData:
    """Sliding windows of closing prices with the date of the forecast target."""

    ticker: str
    window: int
    x: np.ndarray            # (n_samples, window) raw prices
    y: np.ndarray            # (n_samples,) next close
    target_dates: np.ndarray # (n_samples,) date of y
    dates: np.ndarray        # (n_samples,) date of the last price inside x


@dataclass
class Split:
    """One (ticker, regime, window) split with everything a reviewer needs."""

    ticker: str
    regime: str
    window: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_dates: tuple
    val_dates: tuple
    test_dates: tuple
    scaler_x: MinMaxScaler
    scaler_y: MinMaxScaler
    scaler_audit: dict = field(default_factory=dict)
    note: str = ""

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "regime": self.regime,
            "window": self.window,
            "n_train": int(len(self.train)),
            "n_val": int(len(self.val)),
            "n_test": int(len(self.test)),
            "train_start": str(self.train_dates[0]) if len(self.train) else "",
            "train_end": str(self.train_dates[1]) if len(self.train) else "",
            "val_start": str(self.val_dates[0]) if len(self.val) else "",
            "val_end": str(self.val_dates[1]) if len(self.val) else "",
            "test_start": str(self.test_dates[0]) if len(self.test) else "",
            "test_end": str(self.test_dates[1]) if len(self.test) else "",
            **self.scaler_audit,
            "note": self.note,
        }


def make_windows(series: PriceSeries, window: int) -> WindowData:
    """Windows of the last ``window`` closes, target = the next close.

    The target is the close of the day after the window, so a forecast made
    on the last day of the window is evaluated against the next day only.
    """
    closes = series.close
    dates = series.dates
    n = len(closes)
    if n <= window:
        raise ValueError(f"{series.ticker}: {n} rows is not enough for window {window}")
    x = np.lib.stride_tricks.sliding_window_view(closes, window)[: n - window]
    y = closes[window:]
    window_end_dates = dates[window - 1: n - 1]
    target_dates = dates[window:]
    return WindowData(series.ticker, window, np.ascontiguousarray(x),
                      np.ascontiguousarray(y), target_dates, window_end_dates)


def legacy_test_block(series: PriceSeries, regime: str, window: int = 30):
    """The test block the archived runs actually used, expressed as dates.

    The archived code chose the block by index (``train_size = 905 + kk``,
    ``test_size = 100``).  The same block is recovered here from the data so
    that the new run is comparable with the old numbers.
    """
    train_size = config.LEGACY_TRAIN_SIZE[regime]
    test_size = config.LEGACY_TEST_SIZE
    n_windows = len(series.close) - window
    start = train_size
    stop = min(train_size + test_size, n_windows)
    if start >= n_windows:
        return None
    target_dates = series.dates[window:][start:stop]
    return target_dates


def regime_test_dates(series: PriceSeries, regime: str, window: int = 30,
                      mode: str = "legacy") -> np.ndarray | None:
    """Target dates of the test block, per the requested regime definition.

    ``mode='legacy'``   the block the archived runs used (for comparability)
    ``mode='dates'``    the operational regime definition in config.py
    """
    if mode == "legacy":
        return legacy_test_block(series, regime, window)
    definition = config.REGIME_DEFINITION[regime]
    start = pd.Timestamp(definition["start"])
    end = pd.Timestamp(definition["end"])
    target_dates = series.dates[window:]
    mask = (target_dates >= start.to_datetime64()) & (target_dates <= end.to_datetime64())
    return target_dates[mask]


def split_series(series: PriceSeries, regime: str, window: int,
                 val_days: int = config.DEFAULT_VAL_DAYS,
                 regime_mode: str = "legacy",
                 leakage: str = "clean") -> Split:
    """Build a chronological split and fit the scaler on the training part only.

    ``leakage='legacy'`` reproduces the archived behaviour (scaler fitted on
    every window, no validation block) so the size of the leak can be measured
    rather than argued about.
    """
    data = make_windows(series, window)
    target_dates = data.target_dates

    test_dates = regime_test_dates(series, regime, window, regime_mode)
    if test_dates is None or len(test_dates) == 0:
        raise ValueError(f"{series.ticker}/{regime}: no test block found")
    test_mask = np.isin(target_dates, test_dates)
    test_idx = np.where(test_mask)[0]
    if len(test_idx) == 0:
        raise ValueError(f"{series.ticker}/{regime}: test block does not overlap the data")

    first_test = int(test_idx[0])
    if regime_mode == "legacy":
        # the archived block: 100 observations from the legacy anchor
        test_idx = np.arange(first_test,
                              min(first_test + config.LEGACY_TEST_SIZE, len(target_dates)))
    else:
        # the regime's own dates, identical for every window and every model
        test_idx = test_idx

    if leakage == "legacy":
        # the archived split: everything before the test block is training,
        # the test block is also the early-stopping set
        train_idx = np.arange(0, first_test)
        val_idx = np.array([], dtype=int)
    else:
        # clean split: validation sits strictly between training and test
        val_start = max(0, first_test - val_days)
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, first_test)

    x_flat = data.x.reshape(len(data.x), -1)
    if leakage == "legacy":
        scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(x_flat)
        scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(data.y.reshape(-1, 1))
    else:
        # the fix: the scaler only ever sees the training windows
        scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(x_flat[train_idx])
        scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(data.y[train_idx].reshape(-1, 1))

    full_x = MinMaxScaler(feature_range=(0, 1)).fit(x_flat)
    audit = {
        "scaler_x_min_train": float(scaler_x.data_min_.min()),
        "scaler_x_max_train": float(scaler_x.data_max_.max()),
        "scaler_x_min_full": float(full_x.data_min_.min()),
        "scaler_x_max_full": float(full_x.data_max_.max()),
        "scaler_y_min_train": float(scaler_y.data_min_[0]),
        "scaler_y_max_train": float(scaler_y.data_max_[0]),
        "scaler_y_min_full": float(data.y.min()),
        "scaler_y_max_full": float(data.y.max()),
        "leakage_mode": leakage,
    }

    def span(idx):
        if len(idx) == 0:
            return ("", "")
        return (str(pd.Timestamp(target_dates[idx[0]]).date()),
                str(pd.Timestamp(target_dates[idx[-1]]).date()))

    return Split(
        ticker=series.ticker,
        regime=regime,
        window=window,
        train=train_idx,
        val=val_idx,
        test=test_idx,
        train_dates=span(train_idx),
        val_dates=span(val_idx),
        test_dates=span(test_idx),
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        scaler_audit=audit,
        note=("legacy split: scaler fitted on all windows, test block used for "
              "early stopping" if leakage == "legacy" else
              "clean split: scaler fitted on training windows only, validation "
              "block drives early stopping"),
    )


def scaled_arrays(data: WindowData, split: Split):
    """Return (x_train, y_train, x_val, y_val, x_test, y_test) in scaled units.

    The y values are kept in their original price units; the models are fitted
    on the original target as the archived pipeline did, and the scaler for y
    is only used for the generator, which needs a bounded target.
    """
    x_flat = data.x.reshape(len(data.x), -1)
    x_scaled = split.scaler_x.transform(x_flat)
    x_scaled = x_scaled.reshape(len(data.x), data.x.shape[1])
    y_scaled = split.scaler_y.transform(data.y.reshape(-1, 1)).flatten()
    return (x_scaled[split.train], data.y[split.train], y_scaled[split.train],
            x_scaled[split.val], data.y[split.val], y_scaled[split.val],
            x_scaled[split.test], data.y[split.test], y_scaled[split.test])
