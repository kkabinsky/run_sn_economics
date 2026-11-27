"""Loading the archived SET50 price files.

The archived workbooks are Yahoo Finance exports.  Depending on how the file
was saved, the price column is called ``closed``, ``close`` or ``cp``, and the
date column is called ``date``.  The loader accepts all of them and reports
what it found, because the reviewer asked for the exact provenance of every
input (R3.M2).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PRICE_COLUMNS = ["closed", "close", "cp", "Close", "price", "adj close", "Adj Close"]
DATE_COLUMNS = ["date", "Date", "Datetime", "timestamp"]


@dataclass
class PriceSeries:
    """One ticker's daily closing prices, in chronological order."""

    ticker: str
    path: Path
    dates: np.ndarray          # datetime64[ns], ascending
    close: np.ndarray          # float
    price_column: str
    sheet: str
    sha256: str
    open: np.ndarray | None = None    # float, when the workbook has it
    high: np.ndarray | None = None    # float, when the workbook has it
    low: np.ndarray | None = None     # float, when the workbook has it

    def __len__(self) -> int:
        return len(self.close)

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[0])

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[-1])


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def candidate_paths(ticker: str, root: Path | None = None) -> list[Path]:
    """Where the price file of one ticker may live, most authoritative first.

    The copy inside the ticker's own folder is the one the archived training
    script read; the copy in the root of the tree is the raw download and may
    carry a different column name.
    """
    root = root or config.LEGACY_DATA_ROOT
    name = ticker.lower()
    found = []
    for folder in (root / ticker, root):
        for pattern in (f"{name}_final2.xlsx", f"{name}_final.xlsx", f"{name}.xlsx"):
            candidate = folder / pattern
            if candidate.is_file():
                found.append(candidate)
    # keep order, remove duplicates
    unique = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def _pick_column(columns, wanted, kind: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in wanted:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if kind == "date":
        for name, original in lowered.items():
            if "date" in name or "time" in name:
                return original
    return None


def load_price_series(ticker: str, root: Path | None = None) -> PriceSeries:
    """Read one ticker's workbook into a chronological price series."""
    root = root or config.LEGACY_DATA_ROOT
    paths = candidate_paths(ticker, root)
    if not paths:
        raise FileNotFoundError(
            f"no price workbook for {ticker} under {root} "
            f"(looked for {ticker.lower()}_final2.xlsx)")
    path = paths[0]

    workbook = pd.ExcelFile(path)
    frame = None
    sheet_used = None
    price_col = None
    for sheet in workbook.sheet_names:
        if sheet not in ("closed", "close", "Sheet1", "Sheet 1") and sheet != 0:
            continue
        candidate = workbook.parse(sheet)
        found = _pick_column(candidate.columns, PRICE_COLUMNS, "price")
        if found is not None:
            frame = candidate
            sheet_used = sheet
            price_col = found
            break
    if frame is None:                      # last resort: the first sheet
        sheet_used = workbook.sheet_names[0]
        frame = workbook.parse(sheet_used)
        price_col = _pick_column(frame.columns, PRICE_COLUMNS, "price")
    if price_col is None or price_col not in frame.columns:
        raise KeyError(f"{path}: no price column among {PRICE_COLUMNS}")

    date_col = _pick_column(frame.columns, DATE_COLUMNS, "date")
    if date_col is None:
        raise KeyError(f"{path}: no date column among {DATE_COLUMNS}")

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    close = pd.to_numeric(frame[price_col], errors="coerce")
    keep = dates.notna() & close.notna()
    dates = dates[keep]
    close = close[keep]
    order = np.argsort(dates.to_numpy())
    dates = dates.iloc[order]
    close = close.iloc[order]
    dates = dates.reset_index(drop=True)
    close = close.reset_index(drop=True)

    # drop duplicated trading days (keep the last quote of the day)
    frame = pd.DataFrame({"date": dates, "close": close})
    frame = frame.drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    # the open, high and low, when the workbook has them: the money management
    # needs the daily range to place the stop and the target
    extra = {}
    for name in ("open", "high", "low"):
        column = _pick_column(frame.columns, [name, name.upper(), name.capitalize()], name)
        if column is None:
            extra[name] = None
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.reindex(frame.index).fillna(frame["close"])
        extra[name] = values.to_numpy(dtype=float)

    return PriceSeries(
        ticker=ticker,
        path=path,
        dates=frame["date"].to_numpy(),
        close=frame["close"].to_numpy(dtype=float),
        price_column=str(price_col),
        sheet=str(sheet_used),
        sha256=file_sha256(path),
        open=extra.get("open"),
        high=extra.get("high"),
        low=extra.get("low"),
    )


def load_all(tickers=None, root: Path | None = None) -> dict[str, PriceSeries]:
    tickers = tickers or config.TICKERS
    series = {}
    for ticker in tickers:
        try:
            series[ticker] = load_price_series(ticker, root)
        except Exception as error:                     # keep going, report at the end
            print(f"  ! {ticker}: {error}")
    return series
