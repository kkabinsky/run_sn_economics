"""Reproduce the published DM summary matrix and compare the conventions.

The manuscript's Table 2 comes from
``C:\\program_stock\\dmtest_summary_matrix.xlsx``, whose per-stock sheets hold
a 3x3 matrix of z-statistics (CatBoost W5/W10/W30 against LightGBM
W5/W10/W30).  This script rebuilds that matrix from the archived
``backtest_results_iter0.xlsx`` files and prints, for each cell, the three
conventions discussed in the reviewer response:

    z_paper    the published convention (population variance, N(0,1))
    z_matlab   the convention of dmtest.m (sample variance, N(0,1))
    dm_hln     the corrected statistic (Newey-West Bartlett, t(T-1))

A cell that reproduces exactly is the worked example R3.M4 asks for.

    python verify_dm_convention.py --stocks ADVANC PTTGC --publish
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leakfree.dm import dm_statistic, paper_matrix_cell_errors

BASE = Path(r"C:\program_stock")
PUBLISHED = BASE / "dmtest_summary_matrix.xlsx"
CB_VARIANTS = {"W5": BASE / "catboost_window5_crash_new" / "SET50",
               "W10": BASE / "catboost_window10_crash_new" / "SET50",
               "W30": BASE / "catboost_window30_crash_new" / "SET50"}
LGBM_VARIANTS = {"W5": BASE / "lightGMB_window5_crash_new" / "SET50",
                 "W10": BASE / "lightGMB_window10_crash_new" / "SET50",
                 "W30": BASE / "lightGMB_window30_crash_new" / "SET50"}
ROWS = ["CatBoost Window 5", "CatBoost Window 10", "CatBoost Window 30"]
COLS = ["LightGBM Window 5", "LightGBM Window 10", "LightGBM Window 30"]


def stock_matrix(stock: str) -> pd.DataFrame:
    """The 3x3 matrix of the three conventions for one stock."""
    out = pd.DataFrame(index=ROWS, columns=COLS, dtype=float)
    for cb_key in ("W5", "W10", "W30"):
        for lg_key in ("W5", "W10", "W30"):
            cb_file = CB_VARIANTS[cb_key] / stock / "backtest_results_iter0.xlsx"
            lg_file = LGBM_VARIANTS[lg_key] / stock / "backtest_results_iter0.xlsx"
            e1, e2 = paper_matrix_cell_errors(cb_file, lg_file)
            if e1 is None or e2 is None:
                continue
            out.loc[f"CatBoost Window {cb_key[1:]}", f"LightGBM Window {lg_key[1:]}"] = \
                dm_statistic(e1, e2, h=1)["z_paper"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stocks", nargs="*", default=["ADVANC"])
    parser.add_argument("--publish", action="store_true",
                        help="compare with the published workbook")
    args = parser.parse_args()

    published = None
    if args.publish and PUBLISHED.is_file():
        workbook = pd.ExcelFile(PUBLISHED)
        published = {name: workbook.parse(name) for name in workbook.sheet_names
                     if name.lower() not in ("summary", "average dm test", "full result")}
        print(f"published workbook: {PUBLISHED} ({len(published)} stock sheets)")
    else:
        print("published workbook not compared")

    for stock in args.stocks:
        print(f"\n=== {stock} ===")
        mine = stock_matrix(stock)
        print("rebuilt with the published convention (z_paper):")
        print(mine.round(6).to_string())

        if published and stock in published:
            theirs = published[stock]
            # the published sheets keep the row labels in the first column
            if "Unnamed: 0" in theirs.columns:
                theirs = theirs.set_index("Unnamed: 0")
            theirs.index = [str(i).strip() for i in theirs.index]
            theirs = theirs.reindex(index=ROWS, columns=COLS)
            difference = (mine - theirs).abs()
            print("\npublished matrix:")
            print(theirs.round(6).to_string())
            values = difference.to_numpy()
            worst = np.nanmax(values) if np.isfinite(values).any() else float("nan")
            print(f"\nlargest absolute difference: {worst:.3e}")
            print("exact reproduction:", bool(np.isfinite(worst) and worst < 1e-9))

        # the three conventions for one cell
        cb_file = CB_VARIANTS["W30"] / stock / "backtest_results_iter0.xlsx"
        lg_file = LGBM_VARIANTS["W30"] / stock / "backtest_results_iter0.xlsx"
        e1, e2 = paper_matrix_cell_errors(cb_file, lg_file)
        if e1 is not None and e2 is not None:
            result = dm_statistic(e1, e2, h=1)
            print("\nworked example, CatBoost W30 against LightGBM W30:")
            for key, value in result.items():
                print(f"  {key:18} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
