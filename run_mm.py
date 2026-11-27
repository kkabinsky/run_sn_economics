"""Run the sniper money management over the forecasts of a run.

For every regime, window and model the script takes the stored forecasts, runs
the money management of ``leakfree/mm.py`` over the 39 stocks, and compares it
with the always-on signal of the backtest (the equal-weight portfolio).

    python run_mm.py --run outputs/run_20261126_060638 --regimes crash normal \
           --windows 5 10 30 --models CatBoost cGAN
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from leakfree import config as cfg                      # noqa: E402
from leakfree import data as data_mod                  # noqa: E402
from leakfree.mm import SniperRules, run_sniper_mm     # noqa: E402
from leakfree.portfolio import portfolio_frame_from_predictions   # noqa: E402


def load_forecasts(run_dir: Path) -> dict:
    """regime -> window -> model -> ticker -> series of forecasts."""
    out: dict = {}
    folder = run_dir / "data" / "predictions"
    for path in sorted(folder.glob("*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        ticker, regime, window = parts[0], parts[1], int(parts[2].lstrip("W"))
        frame = pd.read_csv(path)
        for model in frame.columns:
            if model in ("date", "actual"):
                continue
            series = pd.Series(frame[model].to_numpy(dtype=float),
                               index=pd.to_datetime(frame["date"]))
            out.setdefault(regime, {}).setdefault(window, {}).setdefault(model, {})[ticker] = series
    return out


def add_hybrids(forecasts: dict) -> dict:
    """The static hybrid of the paper: 0.5 * CatBoost + 0.5 * conGANs.

    The archived pipeline builds it as the simple average of the two forecasts
    (catBoost_conGAN.py, line 765).  Both forecasts live in the same space, so
    the average can be taken from the stored forecasts without training again.
    """
    for regime, windows in forecasts.items():
        for window, models in windows.items():
            catboost = models.get("CatBoost")
            gan = models.get("cGAN")
            if not catboost or not gan:
                continue
            hybrid = {}
            for ticker in set(catboost) & set(gan):
                first = catboost[ticker]
                second = gan[ticker]
                common = first.index.intersection(second.index)
                if len(common) == 0:
                    continue
                hybrid[ticker] = pd.Series(
                    (0.5 * first.loc[common].to_numpy()
                     + 0.5 * second.loc[common].to_numpy()),
                    index=common)
            if hybrid:
                models["Static_Hybrid"] = hybrid
    return forecasts


def prediction_frames(run_dir: Path, forecasts: dict) -> dict:
    """The stored forecasts per cell, in the form the portfolio uses.

    The static hybrid is not stored in the run, it is built from the CatBoost
    and the conGANs forecast, so it is added here as a column of the same
    frame and gets its always-on baseline from the same engine.
    """
    frames: dict = {}
    directory = run_dir / "data" / "predictions"
    for path in sorted(directory.glob("*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        ticker, regime, window = parts[0], parts[1], int(parts[2].lstrip("W"))
        frame = pd.read_csv(path)
        if "CatBoost" in frame.columns and "cGAN" in frame.columns:
            frame = frame.copy()
            frame["Static_Hybrid"] = 0.5 * frame["CatBoost"] + 0.5 * frame["cGAN"]
        frames[(ticker, regime, window)] = frame
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--regimes", nargs="*", default=["crash", "normal"])
    parser.add_argument("--windows", nargs="*", type=int, default=[5, 10, 30])
    parser.add_argument("--models", nargs="*", default=["CatBoost", "cGAN"])
    parser.add_argument("--data-root", default=None, help="where the price workbooks are")
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--target-atr", type=float, default=3.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--edge-atr", type=float, default=0.5)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT / run_dir
    out_dir = Path(args.out) if args.out else run_dir / "mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root) if args.data_root else cfg.LEGACY_DATA_ROOT
    print(f"  prices: {root}")
    prices = data_mod.load_all(root=root)
    print(f"  {len(prices)} price series")

    forecasts = add_hybrids(load_forecasts(run_dir))
    rules = SniperRules(risk_per_trade=args.risk, stop_atr=args.stop_atr,
                         target_atr=args.target_atr, max_positions=args.max_positions,
                         edge_atr=args.edge_atr, cost_bps=args.cost_bps)

    rows = []
    trades_all = []
    for regime in args.regimes:
        for window in args.windows:
            for model in args.models:
                cells = forecasts.get(regime, {}).get(window, {}).get(model)
                if not cells:
                    continue
                result = run_sniper_mm(prices, cells, rules)
                metrics = result["metrics"]
                if not metrics:
                    continue
                metrics.update({"regime": regime, "window": window, "model": model,
                                "stocks": len(cells)})
                rows.append(metrics)
                trades = result["trades"].copy()
                if len(trades):
                    trades["regime"] = regime
                    trades["window"] = window
                    trades["model"] = model
                    trades_all.append(trades)
                print(f"  {regime:7} W{window:<3} {model:10} "
                      f"return {metrics['total_return']*100:8.2f}%  "
                      f"sharpe {metrics['sharpe']:6.2f}  dd {metrics['max_drawdown']*100:7.2f}%  "
                      f"trades {metrics['trades']:4d}  win {metrics['win_rate']*100:5.1f}%  "
                      f"avgR {metrics['average_r']:5.2f}")

    if not rows:
        print("  ไม่มีช่องให้รัน")
        return 1

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "mm_metrics.csv", index=False)
    (out_dir / "rules.json").write_text(
        json.dumps(rules.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(out_dir / "mm_trades.csv", index=False)

    # the always-on signal of the backtest, for the same cells, including the
    # static hybrid, which is not stored in the run
    baseline = portfolio_frame_from_predictions(prediction_frames(run_dir, forecasts),
                                                cost_bps=int(args.cost_bps))
    if len(baseline):
        baseline.to_csv(out_dir / "baseline_portfolio.csv", index=False)
        merged = summary.merge(
            baseline[["regime", "window", "model", "portfolio_return", "sharpe",
                      "ci_low", "ci_high"]].rename(columns={
                          "portfolio_return": "always_on_return",
                          "sharpe": "always_on_sharpe"}),
            on=["regime", "window", "model"], how="left")
        merged["difference"] = merged["total_return"] - merged["always_on_return"]
        merged.to_csv(out_dir / "mm_vs_baseline.csv", index=False)
        merged.to_json(out_dir / "mm_vs_baseline.json", orient="records", indent=2)
        print()
        for _, row in merged.iterrows():
            print(f"  {row['regime']:7} W{int(row['window']):<3} {row['model']:10} "
                  f"sniper {row['total_return']*100:8.2f}%  vs  always-on "
                  f"{row['always_on_return']*100:8.2f}%  "
                  f"[{row['ci_low']*100:7.2f}%, {row['ci_high']*100:7.2f}%]")

    print(f"\n  -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
