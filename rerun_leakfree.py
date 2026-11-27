"""Leakage-free re-run of the SET50 forecasting experiments.

This program answers reviewer comment R3.M2 ("rerun the experiments using
strictly chronological splits and training-only preprocessing, with an
untouched final test set and an auditable replication package") and R3.M3
("provide a stock-by-stock and model-by-model chronology ... recomputed on
genuinely comparable dates and verified regime labels").

It is built on the corrected evaluation engine of the Stage 1 audit
(``leakfree/evaluation.py``, a byte-identical copy of
``D:\\run_revised_conGANs\\revised_core.py``) and it retrains the models with
the three defects of the archived pipeline removed:

    archived :153-160   scaler fitted on every window before the split
    archived :165-170   train / test only, no validation block
    archived :234-240   early stopping on the test set

With ``--db`` the results, the chronology and a live progress table are
written to PostgreSQL, so the run can be watched on the server while it goes.

Example
-------
    python rerun_leakfree.py --tickers ADVANC PTTGC --regimes crash --windows 30
    python rerun_leakfree.py --with-legacy --regimes crash normal --windows 5 10 30
    python rerun_leakfree.py --db postgresql://user:pass@127.0.0.1:15432/forecast_db
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leakfree import config, data as data_mod, db as db_mod, dm as dm_mod
from leakfree import models as models_mod, report as report_mod
from leakfree.evaluation import backtest, buy_and_hold, holm
from leakfree.splits import make_windows, split_series

COMPARATORS = ["CatBoost", "LightGBM", "XGBoost", "RandomWalk"]


def _scaled(data, split):
    """Scaled train / validation / test arrays (see leakfree.splits)."""
    x_flat = data.x.reshape(len(data.x), -1)
    x_scaled = split.scaler_x.transform(x_flat).reshape(data.x.shape)
    y_scaled = split.scaler_y.transform(data.y.reshape(-1, 1)).flatten()
    return (x_scaled[split.train], data.y[split.train], y_scaled[split.train],
            x_scaled[split.val], data.y[split.val], y_scaled[split.val],
            x_scaled[split.test], data.y[split.test], y_scaled[split.test])


def run_cell(series, regime, window, model_names, val_days, regime_mode,
             leakage, seed, cgan_epochs, cgan_batch):
    """Train the requested models on one cell and forecast the test block."""
    split = split_series(series, regime, window, val_days=val_days,
                          regime_mode=regime_mode, leakage=leakage)
    data = make_windows(series, window)
    x_tr, y_tr, ys_tr, x_va, y_va, ys_va, x_te, y_te, ys_te = _scaled(data, split)

    if leakage == "legacy":
        # the archived behaviour: the test block drives early stopping
        early_x, early_y = x_te, y_te
        early_ys = ys_te
    else:
        early_x, early_y = x_va, y_va
        early_ys = ys_va
        if len(early_x) == 0:
            raise ValueError("validation block is empty")

    predictions = {}
    info = {}
    for name in model_names:
        started = time.time()
        if name == "CatBoost":
            model, extra = models_mod.train_catboost(x_tr, y_tr, early_x, early_y, seed)
            predictions[name] = np.asarray(model.predict(x_te), dtype=float)
        elif name == "LightGBM":
            model, extra = models_mod.train_lightgbm(x_tr, y_tr, early_x, early_y, seed)
            predictions[name] = np.asarray(model.predict(x_te), dtype=float)
        elif name == "XGBoost":
            model, extra = models_mod.train_xgboost(x_tr, y_tr, early_x, early_y, seed)
            predictions[name] = np.asarray(model.predict(x_te), dtype=float)
        elif name == "cGAN":
            generator, extra = models_mod.train_cgan(
                x_tr, ys_tr, early_x, early_ys, window=window, seed=seed,
                epochs=cgan_epochs, batch_size=cgan_batch)
            scaled = models_mod.predict_cgan(generator, x_te, seed=seed)
            predictions[name] = split.scaler_y.inverse_transform(
                scaled.reshape(-1, 1)).flatten()
        elif name == "RandomWalk":
            extra = {}
            predictions[name] = models_mod.random_walk_prediction(data, split.test)
        else:
            raise ValueError(f"unknown model {name}")
        extra["seconds"] = round(time.time() - started, 2)
        info[name] = extra

    return split, data, predictions, info


def dm_table(prediction_frames) -> pd.DataFrame:
    """Diebold-Mariano per stock, in all three conventions."""
    rows = []
    for (ticker, regime, window), frame in prediction_frames.items():
        actual = frame["actual"].to_numpy(float)
        for model_a in ["cGAN"] + COMPARATORS:
            if model_a not in frame.columns:
                continue
            error_a = actual - frame[model_a].to_numpy(float)
            for model_b in COMPARATORS:
                if model_b not in frame.columns or model_b == model_a:
                    continue
                error_b = actual - frame[model_b].to_numpy(float)
                if len(error_a) < 5:
                    continue
                result = dm_mod.dm_statistic(error_a, error_b, h=1)
                rows.append({"regime": regime, "window": window, "ticker": ticker,
                             "model_A": model_a, "model_B": model_b, **result})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = np.nan
    for _, group in frame.groupby(["regime", "window", "model_A", "model_B"]):
        frame.loc[group.index, "p_holm"] = holm(group["p_hln"].to_numpy())
    frame["significant_holm"] = frame["p_holm"] < 0.05
    return frame


def paper_table2(prediction_frames) -> list[dict]:
    """The published matrix, rebuilt on genuinely comparable dates.

    The published Table 2 compares CatBoost at one window with LightGBM at
    another, aligning the two error series by position.  Here the two series
    are aligned by date, and the stock is only used when both series cover
    the same days.
    """
    rows = []
    cells = {}
    for (ticker, regime, window), frame in prediction_frames.items():
        for model in COMPARATORS + ["cGAN"]:
            if model not in frame.columns:
                continue
            for (other_ticker, other_regime, other_window), other_frame in prediction_frames.items():
                if other_regime != regime or other_ticker != ticker:
                    continue
                if other_window == window:
                    continue
                for other_model in COMPARATORS + ["cGAN"]:
                    if other_model not in other_frame.columns or other_model == model:
                        continue
                    # one entry per stock: the key must include the ticker, otherwise
                    # only the first stock of every cell is kept
                    key = (regime, window, other_window, model, other_model, ticker)
                    if key in cells:
                        continue
                    left = frame[["date", "actual", model]].rename(
                        columns={model: "pred_a"}).set_index("date")
                    right = other_frame[["date", "actual", other_model]].rename(
                        columns={other_model: "pred_b"}).set_index("date")
                    joined = left.join(right, how="inner", rsuffix="_b")
                    joined = joined.dropna()
                    if len(joined) < 5:
                        continue
                    error_a = (joined["actual"] - joined["pred_a"]).to_numpy(float)
                    error_b = (joined["actual_b"] - joined["pred_b"]).to_numpy(float)
                    cells[key] = (ticker, dm_mod.dm_statistic(error_a, error_b, h=1),
                                   len(joined))
    for key, (ticker, result, n) in cells.items():
        regime, window_a, window_b, model_a, model_b, _ = key
        rows.append({"regime": regime, "window_a": window_a, "window_b": window_b,
                     "model_a": model_a, "model_b": model_b, "ticker": ticker,
                     "n": n, **result})
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = []
    for key, group in frame.groupby(["regime", "window_a", "window_b", "model_a", "model_b"]):
        # dm_statistic gives unadjusted p-values only; the Holm adjustment is made
        # here, over the stocks of the cell (the run stopped on a KeyError before)
        adjusted = holm(group["p_hln"].to_numpy(float))
        grouped.append({
            "regime": key[0], "window_a": key[1], "window_b": key[2],
            "model_a": key[3], "model_b": key[4],
            "n_stocks": int(group["ticker"].nunique()),
            "mean_z_paper": float(group["z_paper"].mean()),
            "mean_z_matlab": float(group["z_matlab"].mean()),
            "mean_dm_hln": float(group["dm_hln"].mean()),
            "mean_p_hln": float(group["p_hln"].mean()),
            "n_significant_holm": int(np.nansum(adjusted < 0.05)),
        })
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--regimes", nargs="*", default=["crash", "normal"],
                        choices=["crash", "normal"])
    parser.add_argument("--windows", nargs="*", type=int, default=config.WINDOWS)
    parser.add_argument("--models", nargs="*", default=config.TRAINED_MODELS)
    parser.add_argument("--val-days", type=int, default=config.DEFAULT_VAL_DAYS)
    parser.add_argument("--regime-mode", default="dates",
                        choices=["dates", "legacy"])
    parser.add_argument("--with-legacy", action="store_true")
    parser.add_argument("--cgan-epochs", type=int, default=config.DEFAULT_CGAN_EPOCHS)
    parser.add_argument("--cgan-batch", type=int, default=config.DEFAULT_CGAN_BATCH)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--db", default=None,
                        help="PostgreSQL URL for the results and the progress table")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    tickers = args.tickers or list(config.TICKERS)
    if args.limit:
        tickers = tickers[: args.limit]
    stamp = datetime.now(timezone.utc)
    run_id = args.run_id or f"run_{stamp:%Y%m%d_%H%M%S}"
    run_dir = Path(args.out) if args.out else config.DEFAULT_OUTPUT_ROOT / f"run_{stamp:%Y%m%d_%H%M%S}"
    if not run_dir.is_absolute():
        run_dir = config.DEFAULT_OUTPUT_ROOT / run_dir
    data_dir = run_dir / "data"
    pred_dir = data_dir / "predictions"
    data_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"run id     : {run_id}")
    print(f"run folder : {run_dir}")
    print(f"tickers={len(tickers)} regimes={args.regimes} windows={args.windows} "
          f"models={args.models} val_days={args.val_days} regime_mode={args.regime_mode}")

    database = None
    database_url = args.db or db_mod.default_url()
    if database_url:
        database = db_mod.Database(database_url)
        database.create_schema()
        print("  connected to PostgreSQL")

    run_config = {
        "tickers": tickers, "regimes": args.regimes, "windows": args.windows,
        "models": args.models, "val_days": args.val_days,
        "regime_mode": args.regime_mode, "with_legacy": args.with_legacy,
        "cost_grid_bps": config.DEFAULT_COST_BPS, "cgan_epochs": args.cgan_epochs,
        "seed": args.seed, "data_root": str(config.LEGACY_DATA_ROOT),
    }
    if database:
        database.start_run(run_id, run_config)

    print("loading price files ...")
    series_map = data_mod.load_all(tickers)
    print(f"  {len(series_map)} price series available")

    chronology_rows = []
    metric_rows = []
    prediction_frames = {}
    input_records = []
    failures = []

    for ticker, series in series_map.items():
        input_records.append({
            "ticker": ticker, "path": str(series.path), "sha256": series.sha256,
            "price_column": series.price_column, "sheet": series.sheet,
            "rows": len(series), "first_date": str(series.first_date.date()),
            "last_date": str(series.last_date.date()),
        })
        if database:
            database.upsert_stock(ticker, series.dates, series.close,
                                  str(series.path), series.sha256)
        for regime in args.regimes:
            for window in args.windows:
                tag = f"{ticker} {regime} W{window}"
                if database:
                    database.mark_progress(run_id, ticker, regime, window, "running")
                try:
                    split, data, predictions, info = run_cell(
                        series, regime, window, args.models, args.val_days,
                        args.regime_mode, "clean", args.seed,
                        args.cgan_epochs, args.cgan_batch)
                except Exception as error:
                    print(f"  ! {tag}: {type(error).__name__}: {error}")
                    failures.append({"ticker": ticker, "regime": regime,
                                     "window": window, "error": str(error)})
                    if database:
                        database.mark_progress(run_id, ticker, regime, window, "failed",
                                               message=f"{type(error).__name__}: {error}")
                    continue

                summary = split.summary()
                summary["models"] = ";".join(args.models)
                summary["model_info"] = json.dumps(info, default=str)
                summary["source_file"] = str(series.path)
                summary["source_sha256"] = series.sha256
                summary["leakage"] = "clean"
                chronology_rows.append(summary)

                frame = pd.DataFrame({
                    "date": pd.to_datetime(data.target_dates[split.test]).date,
                    "actual": data.y[split.test],
                })
                for name, values in predictions.items():
                    frame[name] = values
                prediction_frames[(ticker, regime, window)] = frame
                frame.to_csv(pred_dir / f"{ticker}_{regime}_W{window}.csv", index=False)

                for name, values in predictions.items():
                    for cost in config.DEFAULT_COST_BPS:
                        result = backtest(data.y[split.test], values, cost_bps=cost)
                        if result is None:
                            continue
                        metric_rows.append({
                            "ticker": ticker, "regime": regime, "window": window,
                            "model": name, "cost_bps": cost, "leakage": "clean",
                            **{k: v for k, v in result.items()
                               if k not in ("equity", "net_returns", "positions")},
                        })
                print(f"  ok {tag}: train={summary['n_train']} val={summary['n_val']} "
                      f"test={summary['n_test']} ({summary['test_start']}..{summary['test_end']})")

                if database:
                    database.mark_progress(run_id, ticker, regime, window, "done",
                                           n_train=summary["n_train"],
                                           n_val=summary["n_val"],
                                           n_test=summary["n_test"],
                                           test_start=summary["test_start"],
                                           test_end=summary["test_end"])
                    database.insert_chronology(run_id, summary)
                    database.insert_predictions(run_id, ticker, regime, window, frame)

                if args.with_legacy:
                    try:
                        lsplit, ldata, lpredictions, _ = run_cell(
                            series, regime, window, args.models, args.val_days,
                            "legacy", "legacy", args.seed,
                            args.cgan_epochs, args.cgan_batch)
                        for name, values in lpredictions.items():
                            for cost in config.DEFAULT_COST_BPS:
                                result = backtest(ldata.y[lsplit.test], values, cost_bps=cost)
                                if result is None:
                                    continue
                                metric_rows.append({
                                    "ticker": ticker, "regime": regime, "window": window,
                                    "model": name, "cost_bps": cost, "leakage": "legacy",
                                    **{k: v for k, v in result.items()
                                       if k not in ("equity", "net_returns", "positions")},
                                })
                    except Exception as error:
                        print(f"    (legacy rule failed: {error})")

    chronology = pd.DataFrame(chronology_rows)
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics.to_csv(data_dir / "metrics_per_stock.csv", index=False)
    clean = metrics[metrics["leakage"] == "clean"] if not metrics.empty else metrics
    legacy = metrics[metrics["leakage"] == "legacy"] if not metrics.empty else metrics

    dm = dm_table(prediction_frames)
    if not dm.empty:
        dm.to_csv(data_dir / "dm_per_stock.csv", index=False)
    table2 = paper_table2(prediction_frames)
    if table2:
        pd.DataFrame(table2).to_csv(data_dir / "paper_table2.csv", index=False)

    coverage = report_mod.coverage_table(chronology, prediction_frames)
    if not coverage.empty:
        coverage.to_csv(data_dir / "coverage.csv", index=False)
    label_check = report_mod.regime_label_check(chronology)
    if not label_check.empty:
        label_check.to_csv(data_dir / "regime_label_check.csv", index=False)
    leakage = report_mod.leakage_effect_table(clean, legacy)
    if not leakage.empty:
        leakage.to_csv(data_dir / "leakage_effect.csv", index=False)
    if not chronology.empty:
        chronology.to_csv(data_dir / "chronology.csv", index=False)

    if database:
        database.insert_metrics(run_id, clean)
        database.insert_dm(run_id, dm)
        database.insert_paper_table2(run_id, table2)
        database.finish_run(run_id, "finished",
                             notes=f"{len(chronology_rows)} cells, {len(failures)} failures")

    report_mod.write_provenance(run_dir, run_config, input_records)
    report_mod.build_report(run_dir, run_config, chronology, clean, dm,
                            coverage, leakage, label_check)
    if failures:
        (data_dir / "failures.json").write_text(json.dumps(failures, indent=2,
                                                           ensure_ascii=False), encoding="utf-8")
    report_mod.write_manifest(run_dir)
    if database:
        database.close()
    print(f"done -> {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
