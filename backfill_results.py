"""Fill the result tables of a finished run in the database.

The runner writes the metrics, the DM tests and the paper matrix at the end of
a run.  When the SSH tunnel to the server drops in that last step, the cells
are already in the database but the summary tables are not.  This script
re-reads the CSV files of the run folder and writes them, so the database and
the response letter always carry the same numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from leakfree import db as db_mod   # noqa: E402


def backfill(run_id: str, run_dir: Path, database) -> dict:
    data = run_dir / "data"
    written = {}

    metrics_path = data / "metrics_per_stock.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        if len(metrics):
            clean = metrics[metrics["leakage"] == "clean"].copy()
            legacy = metrics[metrics["leakage"] == "legacy"].copy()
            database.delete_metrics(run_id)
            database.insert_metrics(run_id, clean)
            if len(legacy):
                database.insert_metrics(run_id, legacy)
            written["metrics_clean"] = int(len(clean))
            written["metrics_legacy"] = int(len(legacy))

    dm_path = data / "dm_per_stock.csv"
    if dm_path.is_file():
        dm = pd.read_csv(dm_path)
        if len(dm):
            database.delete_dm(run_id)
            database.insert_dm(run_id, dm)
            written["dm"] = int(len(dm))

    table2_path = data / "paper_table2.csv"
    if table2_path.is_file():
        table2 = pd.read_csv(table2_path)
        if len(table2):
            database.delete_paper_table2(run_id)
            database.insert_paper_table2(run_id, table2.to_dict("records"))
            written["paper_table2"] = int(len(table2))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run", required=True, help="the run folder")
    parser.add_argument("--db", default=None)
    parser.add_argument("--status", default="finished")
    args = parser.parse_args()

    url = args.db or os.environ.get("FORECAST_DATABASE_URL")
    if not url:
        print("no database url")
        return 1
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT / run_dir

    database = db_mod.Database(url)
    database.create_schema()
    written = backfill(args.run_id, run_dir, database)
    database.finish_run(args.run_id, args.status)
    print(json.dumps({"run_id": args.run_id, "written": written}, ensure_ascii=False))
    database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
