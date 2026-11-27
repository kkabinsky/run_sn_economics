"""Check that a fresh run still reproduces the results stored in the repository.

The workflow re-runs one cell (ADVANC, crash, W5) and this script compares the
result with the row that is stored in ``output/data/metrics_per_stock.csv``.
The comparison is made on the model that does not depend on the platform
(random walk) exactly, and on the trained model with a tolerance, because the
gradient boosting libraries do not produce bit-identical models on different
operating systems.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOLERANCE = 1e-3      # relative tolerance for the trained model
EXACT = 1e-12         # the random walk has no training, it must match exactly
KEY = ["ticker", "regime", "window", "model", "cost_bps"]


def main() -> int:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/smoke")
    fresh = pd.read_csv(run / "data" / "metrics_per_stock.csv")
    stored = pd.read_csv("output/data/metrics_per_stock.csv")

    fresh = fresh[(fresh.leakage == "clean") & (fresh.cost_bps == 20)]
    stored = stored[(stored.leakage == "clean") & (stored.cost_bps == 20)
                    & (stored.ticker == "ADVANC") & (stored.regime == "crash")
                    & (stored.window == 5)]
    merged = fresh.merge(stored, on=KEY, suffixes=("_new", "_old"))
    if merged.empty:
        print("no rows to compare")
        return 1

    failures = []
    for _, row in merged.iterrows():
        tolerance = EXACT if row["model"] == "RandomWalk" else TOLERANCE
        for column in ("rmse", "mase", "cum_return_geom", "hit_rate"):
            new = float(row[column + "_new"])
            old = float(row[column + "_old"])
            scale = max(abs(old), 1e-12)
            difference = abs(new - old) / scale
            status = "ok" if difference <= tolerance else "DIFFERS"
            print(f"  {status:8} {row['model']:11} {column:16} "
                  f"{new:.6f} vs {old:.6f}  (relative {difference:.2e})")
            if difference > tolerance:
                failures.append(f"{row['model']} {column}")

    # the chronology has to be identical, it does not depend on the platform
    new_chronology = pd.read_csv(run / "data" / "chronology.csv")
    stored_chronology = pd.read_csv("output/data/chronology.csv")
    columns = ["train_start", "train_end", "val_start", "val_end", "test_start",
               "test_end", "n_train", "n_val", "n_test"]
    chronology = new_chronology.merge(
        stored_chronology, on=["ticker", "regime", "window"], suffixes=("_new", "_old"))
    for column in columns:
        same = (chronology[column + "_new"].astype(str)
                == chronology[column + "_old"].astype(str)).all()
        print(f"  {'ok' if same else 'DIFFERS':8} chronology {column}")
        if not same:
            failures.append(f"chronology {column}")

    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print("\nOK: the fresh run reproduces the stored result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
