"""Check that every test block lies inside the definition of its regime.

The reviewer asks for the regime labels to be verified.  The definitions are
the ones of the manuscript's own group table:

    crash   2020-02-17 to 2020-03-30
    normal  2023-11-08 to 2024-04-03

The script reads the chronology of the run and fails when a cell forecasts
outside the interval that defines its regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFINITIONS = {
    "crash": ("2020-02-17", "2020-03-30"),
    "normal": ("2023-11-08", "2024-04-03"),
}


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "output/data/chronology.csv")
    chronology = pd.read_csv(source)
    failures = []
    for regime, (start, end) in DEFINITIONS.items():
        cells = chronology[chronology.regime == regime]
        if cells.empty:
            print(f"  {regime:8} ไม่มีช่องในรอบนี้")
            continue
        inside = ((cells.test_start >= start) & (cells.test_end <= end)).all()
        same_dates = cells.test_start.nunique() == 1 and cells.test_end.nunique() == 1
        print(f"  {regime:8} ช่อง {len(cells):3} | ช่วงทดสอบ "
              f"{cells.test_start.min()} .. {cells.test_end.max()} | "
              f"อยู่ในนิยาม: {inside} | วันเดียวกันทุก window: {same_dates}")
        if not inside:
            failures.append(regime)
        if not same_dates:
            failures.append(regime + " (dates differ between windows)")

    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print("\nOK: every test block lies inside the definition of its regime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
