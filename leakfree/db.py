"""PostgreSQL storage for the re-run: results, chronology and live progress.

The schema keeps the shape of the archived SQLite file
(``D:\\xgboost\\stocks.db``, table ``stock`` with ``datetime_stock, open,
high, low, closed_price, volume``) and adds the tables the replication
package needs:

    runs          one row per run, with its configuration and status
    run_progress  one row per stock x regime x window, updated while the run
                  is going, so the progress table on the server is live
    stock         the price history, per ticker
    chronology    the training / validation / test dates and the normalisation
                  constants fitted on training only
    predictions   the forecast values, per model
    metrics       the backtest metrics per model and cost grid
    dm_tests      the Diebold-Mariano results in all three conventions
    paper_table2  the cross-stock matrix in the layout of the published
                  Table 2, for a direct comparison with the paper

Connection: the ``FORECAST_DATABASE_URL`` environment variable, or the
``--db`` argument of the runner.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock (
    id             serial PRIMARY KEY,
    ticker         text NOT NULL,
    datetime_stock timestamptz NOT NULL,
    open           real,
    high           real,
    low            real,
    closed_price   real NOT NULL,
    volume         real,
    source_file    text,
    source_sha256  text,
    UNIQUE (ticker, datetime_stock)
);
CREATE INDEX IF NOT EXISTS stock_ticker_idx ON stock (ticker, datetime_stock);

CREATE TABLE IF NOT EXISTS runs (
    run_id      text PRIMARY KEY,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status      text NOT NULL DEFAULT 'running',
    regime_mode text,
    val_days    integer,
    seed        integer,
    config      jsonb,
    notes       text
);

CREATE TABLE IF NOT EXISTS run_progress (
    run_id      text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ticker      text NOT NULL,
    regime      text NOT NULL,
    window_size integer NOT NULL,
    status      text NOT NULL DEFAULT 'pending',
    started_at  timestamptz,
    finished_at timestamptz,
    n_train     integer,
    n_val       integer,
    n_test      integer,
    test_start  date,
    test_end    date,
    message     text,
    PRIMARY KEY (run_id, ticker, regime, window_size)
);

CREATE TABLE IF NOT EXISTS chronology (
    run_id      text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ticker      text NOT NULL,
    regime      text NOT NULL,
    window_size integer NOT NULL,
    train_start date,
    train_end   date,
    val_start   date,
    val_end     date,
    test_start  date,
    test_end    date,
    n_train     integer,
    n_val       integer,
    n_test      integer,
    leakage     text,
    scaler_x_min_train double precision,
    scaler_x_max_train double precision,
    scaler_x_min_full  double precision,
    scaler_x_max_full  double precision,
    scaler_y_min_train double precision,
    scaler_y_max_train double precision,
    scaler_y_min_full  double precision,
    scaler_y_max_full  double precision,
    models      text,
    model_info  jsonb,
    source_file text,
    source_sha256 text,
    PRIMARY KEY (run_id, ticker, regime, window_size)
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id    text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ticker    text NOT NULL,
    regime    text NOT NULL,
    window_size integer NOT NULL,
    model     text NOT NULL,
    date      date NOT NULL,
    actual    double precision,
    predicted double precision
);
CREATE INDEX IF NOT EXISTS predictions_cell_idx
    ON predictions (run_id, regime, window_size, ticker, model);

CREATE TABLE IF NOT EXISTS metrics (
    run_id     text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ticker     text NOT NULL,
    regime     text NOT NULL,
    window_size integer NOT NULL,
    model      text NOT NULL,
    cost_bps   integer NOT NULL,
    leakage    text NOT NULL DEFAULT 'clean',
    n_days     integer,
    cum_return_geom double precision,
    cum_return_sum  double precision,
    sharpe_annual    double precision,
    mean_daily       double precision,
    vol_annual       double precision,
    max_drawdown     double precision,
    hit_rate         double precision,
    exposure         double precision,
    turnover_total   double precision,
    n_trades         integer,
    cost_drag        double precision,
    rmse             double precision,
    mae              double precision,
    mase             double precision
);
CREATE INDEX IF NOT EXISTS metrics_cell_idx
    ON metrics (run_id, regime, window_size, model);

CREATE TABLE IF NOT EXISTS dm_tests (
    run_id       text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    regime       text NOT NULL,
    window_size integer NOT NULL,
    ticker       text NOT NULL,
    model_a      text NOT NULL,
    model_b      text NOT NULL,
    z_paper      double precision,
    p_paper_normal double precision,
    z_matlab     double precision,
    p_matlab_normal double precision,
    dm_hln       double precision,
    p_hln        double precision,
    mean_d       double precision,
    n            integer,
    better       text,
    p_holm       double precision,
    significant_holm boolean
);
CREATE INDEX IF NOT EXISTS dm_cell_idx
    ON dm_tests (run_id, regime, window_size, model_a, model_b);

CREATE TABLE IF NOT EXISTS paper_table2 (
    run_id     text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    regime     text NOT NULL,
    window_a   integer NOT NULL,
    window_b   integer NOT NULL,
    model_a    text NOT NULL,
    model_b    text NOT NULL,
    n_stocks   integer,
    mean_z_paper  double precision,
    mean_z_matlab double precision,
    mean_dm_hln   double precision,
    mean_p_hln    double precision,
    n_significant_holm integer
);
"""


def default_url() -> str | None:
    return os.environ.get("FORECAST_DATABASE_URL")


class Database:
    """Thin wrapper over psycopg with the writes the runner needs."""

    def __init__(self, url: str | None = None):
        self.url = url or default_url()
        if not self.url:
            raise RuntimeError("no database URL: set FORECAST_DATABASE_URL or pass --db")
        self.connection = psycopg.connect(self.url, autocommit=False)

    # ------------------------------------------------------------------ setup
    def create_schema(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass

    # ------------------------------------------------------------------- runs
    def start_run(self, run_id: str, config: dict, notes: str = "") -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO runs (run_id, status, regime_mode, val_days, seed, config, notes)
                   VALUES (%s, 'running', %s, %s, %s, %s, %s)
                   ON CONFLICT (run_id) DO UPDATE SET status = 'running',
                        started_at = now(), config = EXCLUDED.config""",
                (run_id, config.get("regime_mode"), config.get("val_days"),
                 config.get("seed"), json.dumps(config, default=str), notes))
        self.connection.commit()

    def finish_run(self, run_id: str, status: str = "finished", notes: str = "") -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """UPDATE runs SET status = %s, finished_at = now(),
                          notes = COALESCE(NULLIF(%s, ''), notes) WHERE run_id = %s""",
                (status, notes, run_id))
        self.connection.commit()

    # ---------------------------------------------------------------- progress
    def mark_progress(self, run_id: str, ticker: str, regime: str, window: int,
                      status: str, **fields) -> None:
        columns = {"status": status}
        for key in ("n_train", "n_val", "n_test"):
            if key in fields and fields[key] is not None:
                columns[key] = int(fields[key])
        for key in ("test_start", "test_end"):
            if fields.get(key):
                columns[key] = fields[key]
        if fields.get("message"):
            columns["message"] = str(fields["message"])[:500]
        if status == "running":
            columns["started_at"] = datetime.now(timezone.utc)
        if status in ("done", "failed"):
            columns["finished_at"] = datetime.now(timezone.utc)
        assignments = ", ".join(f"{name} = %s" for name in columns)
        values = list(columns.values())
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO run_progress
                       (run_id, ticker, regime, window_size, {', '.join(columns)})
                       VALUES (%s, %s, %s, %s, {', '.join(['%s'] * len(columns))})
                       ON CONFLICT (run_id, ticker, regime, window_size) DO UPDATE
                       SET {assignments}""",
                # the VALUES list and the ON CONFLICT SET list use the same values
                [run_id, ticker, regime, window] + values + values)
        self.connection.commit()

    # ------------------------------------------------------------------- data
    def upsert_stock(self, ticker: str, dates, closes, source_file: str,
                     source_sha256: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO stock (ticker, datetime_stock, closed_price, source_file, source_sha256)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (ticker, datetime_stock) DO UPDATE SET closed_price = EXCLUDED.closed_price""",
                [(ticker, str(date), float(close), source_file, source_sha256)
                 for date, close in zip(dates, closes)])
        self.connection.commit()

    def insert_chronology(self, run_id: str, row: dict) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO chronology (
                       run_id, ticker, regime, window_size, train_start, train_end,
                       val_start, val_end, test_start, test_end, n_train, n_val,
                       n_test, leakage, scaler_x_min_train, scaler_x_max_train,
                       scaler_x_min_full, scaler_x_max_full, scaler_y_min_train,
                       scaler_y_max_train, scaler_y_min_full, scaler_y_max_full,
                       models, model_info, source_file, source_sha256)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (run_id, ticker, regime, window_size) DO UPDATE SET
                       n_test = EXCLUDED.n_test, test_start = EXCLUDED.test_start,
                       test_end = EXCLUDED.test_end""",
                (run_id, row["ticker"], row["regime"], row["window"],
                 row.get("train_start") or None, row.get("train_end") or None,
                 row.get("val_start") or None, row.get("val_end") or None,
                 row.get("test_start") or None, row.get("test_end") or None,
                 row.get("n_train"), row.get("n_val"), row.get("n_test"),
                 row.get("leakage"), row.get("scaler_x_min_train"),
                 row.get("scaler_x_max_train"), row.get("scaler_x_min_full"),
                 row.get("scaler_x_max_full"), row.get("scaler_y_min_train"),
                 row.get("scaler_y_max_train"), row.get("scaler_y_min_full"),
                 row.get("scaler_y_max_full"), row.get("models"),
                 json.dumps(json.loads(row["model_info"]), default=str)
                 if row.get("model_info") else None,
                 row.get("source_file"), row.get("source_sha256")))
        self.connection.commit()

    def insert_predictions(self, run_id: str, ticker: str, regime: str, window: int,
                           frame) -> None:
        if frame is None or not len(frame):
            return
        model_columns = [c for c in frame.columns if c not in ("date", "actual")]
        rows = []
        for _, row in frame.iterrows():
            date = str(row["date"])
            actual = float(row["actual"])
            for model in model_columns:
                rows.append((run_id, ticker, regime, window, model, date, actual,
                             float(row[model])))
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO predictions
                       (run_id, ticker, regime, window_size, model, date, actual, predicted)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""", rows)
        self.connection.commit()

    def insert_metrics(self, run_id: str, frame) -> None:
        if frame is None or not len(frame):
            return
        columns = ["n_days", "cum_return_geom", "cum_return_sum", "sharpe_annual",
                   "mean_daily", "vol_annual", "max_drawdown", "hit_rate", "exposure",
                   "turnover_total", "n_trades", "cost_drag", "rmse", "mae", "mase"]
        rows = []
        for _, row in frame.iterrows():
            rows.append([run_id, row["ticker"], row["regime"], int(row["window"]),
                         row["model"], int(row["cost_bps"]), row.get("leakage", "clean")]
                        + [row.get(name) for name in columns])
        with self.connection.cursor() as cursor:
            cursor.executemany(
                f"""INSERT INTO metrics
                        (run_id, ticker, regime, window_size, model, cost_bps, leakage,
                         {', '.join(columns)})
                    VALUES ({', '.join(['%s'] * (7 + len(columns)))})""", rows)
        self.connection.commit()

    def insert_dm(self, run_id: str, frame) -> None:
        if frame is None or not len(frame):
            return
        columns = ["z_paper", "p_paper_normal", "z_matlab", "p_matlab_normal",
                   "dm_hln", "p_hln", "mean_d", "n", "better", "p_holm", "significant_holm"]
        rows = []
        for _, row in frame.iterrows():
            rows.append([run_id, row["regime"], int(row["window"]), row["ticker"],
                         row["model_A"], row["model_B"]]
                        + [row.get(name) for name in columns])
        with self.connection.cursor() as cursor:
            cursor.executemany(
                f"""INSERT INTO dm_tests
                        (run_id, regime, window_size, ticker, model_a, model_b, {', '.join(columns)})
                    VALUES ({', '.join(['%s'] * (6 + len(columns)))})""", rows)
        self.connection.commit()

    def delete_metrics(self, run_id: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM metrics WHERE run_id = %s", (run_id,))
        self.connection.commit()

    def delete_dm(self, run_id: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM dm_tests WHERE run_id = %s", (run_id,))
        self.connection.commit()

    def delete_paper_table2(self, run_id: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM paper_table2 WHERE run_id = %s", (run_id,))
        self.connection.commit()

    def insert_paper_table2(self, run_id: str, rows: list[dict]) -> None:
        if not rows:
            return
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO paper_table2
                       (run_id, regime, window_a, window_b, model_a, model_b, n_stocks,
                        mean_z_paper, mean_z_matlab, mean_dm_hln, mean_p_hln, n_significant_holm)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(run_id, r["regime"], r["window_a"], r["window_b"], r["model_a"],
                  r["model_b"], r.get("n_stocks"), r.get("mean_z_paper"),
                  r.get("mean_z_matlab"), r.get("mean_dm_hln"), r.get("mean_p_hln"),
                  r.get("n_significant_holm")) for r in rows])
        self.connection.commit()
