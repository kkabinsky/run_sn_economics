"""The analyses behind the answer to reviewer comment R3.M4.

R3.M4 asks, besides the definition of the Diebold-Mariano statistic, for

* a worked example that connects one cell of the test table to its paired losses,
* how forecasts of different regimes can be compared on the same dates,
* the meaning of cross-stock averages of test statistics and the treatment of
  serial and cross-sectional dependence,
* paired sample sizes, mean loss differentials, uncertainty estimates and
  adjusted p-values.

Everything here is computed from the stored forecasts of a run
(``data/predictions/<TICKER>_<regime>_W<window>.csv``, one file per stock,
regime and window, with one ``actual`` column and one column per model) and
from the per-stock test table ``data/dm_per_stock.csv`` of the same run.

Two kinds of test are reported:

per stock
    the test of ``rerun_leakfree.py``: squared price errors, Newey-West variance
    at lag h - 1 = 0, Harvey-Leybourne-Newbold factor, t(T-1) reference, Holm
    adjustment over the stocks of a cell.  As a check on serial dependence it is
    recomputed with ``auto_lag(T)`` Bartlett lags.

pooled
    for each date the loss differential is averaged over the stocks of the
    cell, and the test is run on that daily average.  Because the average is
    formed date by date before the variance is estimated, the covariance
    between stocks enters the variance, so the test allows for
    cross-sectional dependence.  The errors are scaled by the previous close
    (squared return errors, in percent squared), so that stocks with high
    prices do not dominate the average.  Newey-West variance with
    ``auto_lag(T)`` Bartlett lags, Harvey-Leybourne-Newbold factor, t(T-1)
    reference, Holm adjustment over the model pairs of a cell.

Sign convention throughout: d_t = loss_A - loss_B, so a positive mean or
statistic means that model A has the larger loss.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import dm as dm_mod
from .evaluation import holm

# the order in which a pair is written: the first model is A
MODEL_ORDER = ["cGAN", "CatBoost", "LightGBM", "XGBoost", "RandomWalk"]

# the worked example: one stock and one cell of the crash regime
EXAMPLE = {"ticker": "ADVANC", "regime": "crash", "window": 10}


def auto_lag(n: int) -> int:
    """Newey-West rule of thumb, floor(4 (T/100)^(2/9)): 3 for T = 31, 4 for T = 100."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=float)
    n = d.size
    centred = d - d.mean()
    total = float(centred @ centred) / n
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(centred[k:] @ centred[:-k]) / n
        total += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return max(total, 1e-300)


def hln_test(d: np.ndarray, lag: int) -> dict:
    """DM statistic with a Newey-West variance, the HLN factor for h = 1, t(T-1)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 5 or float(np.std(d)) == 0.0:
        return {"n": n, "mean_d": float(np.mean(d)) if n else float("nan"),
                "se": float("nan"), "stat": float("nan"), "p": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "lag": lag}
    se = math.sqrt(newey_west_variance(d, lag) / n)
    stat = float(d.mean()) / se * math.sqrt((n - 1.0) / n)
    p = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
    half = float(stats.t.ppf(0.975, df=n - 1)) * se
    return {"n": n, "mean_d": float(d.mean()), "se": se, "stat": stat, "p": p,
            "ci_low": float(d.mean()) - half, "ci_high": float(d.mean()) + half,
            "lag": lag}


def load_predictions(run_dir: Path) -> dict:
    """{(ticker, regime, window): frame} from data/predictions of a run."""
    frames = {}
    folder = Path(run_dir) / "data" / "predictions"
    if not folder.is_dir():
        return frames
    for path in sorted(folder.glob("*.csv")):
        try:
            ticker, regime, window = path.stem.rsplit("_", 2)
            window = int(window.lstrip("W"))
        except ValueError:
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frames[(ticker, regime, window)] = frame.sort_values("date").reset_index(drop=True)
    return frames


def _models(frames: dict) -> list[str]:
    present = set()
    for frame in frames.values():
        present.update(c for c in frame.columns if c in MODEL_ORDER)
    return [m for m in MODEL_ORDER if m in present]


def _previous_close(frame: pd.DataFrame) -> pd.Series:
    if "RandomWalk" in frame.columns:
        return frame["RandomWalk"].astype(float)
    return frame["actual"].astype(float).shift(1)


def worked_example(frames: dict, dm_table: pd.DataFrame, models: list[str]) -> dict:
    """One cell of the per-stock test table, traced back to its paired losses."""
    key = (EXAMPLE["ticker"], EXAMPLE["regime"], EXAMPLE["window"])
    if key not in frames:
        return {}
    frame = frames[key]
    model_a = "cGAN" if "cGAN" in models else "CatBoost"
    model_b = "CatBoost" if model_a == "cGAN" else "LightGBM"
    if model_a not in frame.columns or model_b not in frame.columns:
        return {}
    paired = frame[["date", "actual", model_a, model_b]].dropna().copy()
    paired["loss_A"] = (paired["actual"] - paired[model_a]) ** 2
    paired["loss_B"] = (paired["actual"] - paired[model_b]) ** 2
    paired["d"] = paired["loss_A"] - paired["loss_B"]
    d = paired["d"].to_numpy(float)
    n = d.size
    result = dm_mod.dm_statistic((paired["actual"] - paired[model_a]).to_numpy(float),
                                 (paired["actual"] - paired[model_b]).to_numpy(float), h=1)
    row = pd.DataFrame()
    if len(dm_table):
        row = dm_table[(dm_table["ticker"] == key[0]) & (dm_table["regime"] == key[1])
                       & (dm_table["window"] == key[2]) & (dm_table["model_A"] == model_a)
                       & (dm_table["model_B"] == model_b)]
    lagged = hln_test(d, auto_lag(n))
    family = 0
    if len(dm_table):
        family = int(((dm_table["regime"] == key[1]) & (dm_table["window"] == key[2])
                      & (dm_table["model_A"] == model_a)
                      & (dm_table["model_B"] == model_b)).sum())
    example = {
        "ticker": key[0], "regime": key[1], "window": key[2],
        "model_a": model_a, "model_b": model_b, "n": n,
        "first_date": str(paired["date"].iloc[0].date()),
        "last_date": str(paired["date"].iloc[-1].date()),
        "mse_a": float(paired["loss_A"].mean()), "mse_b": float(paired["loss_B"].mean()),
        "rmse_a": float(math.sqrt(paired["loss_A"].mean())),
        "rmse_b": float(math.sqrt(paired["loss_B"].mean())),
        "mean_d": float(d.mean()),
        "var_pop": float(np.var(d, ddof=0)), "var_sample": float(np.var(d, ddof=1)),
        "z_paper": result["z_paper"], "z_matlab": result["z_matlab"],
        "dm_hln": result["dm_hln"], "p_hln": result["p_hln"],
        "p_holm": float(row["p_holm"].iloc[0]) if len(row) else float("nan"),
        "lag_auto": lagged["lag"], "dm_hln_lag_auto": lagged["stat"],
        "p_lag_auto": lagged["p"],
        "family_size": family,
        "matches_table": bool(len(row) and abs(float(row["z_paper"].iloc[0])
                                                - result["z_paper"]) < 1e-9),
        "rows": [{"date": str(r.date.date()), "actual": float(r.actual),
                  "pred_a": float(getattr(r, model_a)), "pred_b": float(getattr(r, model_b)),
                  "loss_a": float(r.loss_A), "loss_b": float(r.loss_B), "d": float(r.d)}
                 for r in paired.head(5).itertuples(index=False)],
    }
    return example, paired


def pooled_tests(frames: dict, dm_table: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per regime, window and model pair."""
    rows = []
    cells = sorted({(regime, window) for (_, regime, window) in frames})
    for regime, window in cells:
        stock_frames = {ticker: frame for (ticker, r, w), frame in frames.items()
                        if r == regime and w == window}
        for model_a, model_b in combinations(models, 2):
            per_date = []
            per_stock_lag = []
            for ticker, frame in stock_frames.items():
                if model_a not in frame.columns or model_b not in frame.columns:
                    continue
                prev = _previous_close(frame)
                actual = frame["actual"].astype(float)
                err_a = (actual - frame[model_a].astype(float))
                err_b = (actual - frame[model_b].astype(float))
                scaled = ((err_a / prev * 100.0) ** 2 - (err_b / prev * 100.0) ** 2)
                per_date.append(pd.Series(scaled.to_numpy(float),
                                          index=frame["date"], name=ticker))
                d_price = (err_a ** 2 - err_b ** 2).to_numpy(float)
                d_price = d_price[np.isfinite(d_price)]
                if d_price.size >= 5:
                    per_stock_lag.append(hln_test(d_price, auto_lag(d_price.size))["p"])
            if not per_date:
                continue
            panel = pd.concat(per_date, axis=1)
            daily = panel.mean(axis=1, skipna=True).dropna().to_numpy(float)
            t_obs = daily.size
            pooled = hln_test(daily, auto_lag(t_obs))

            ordered = dm_table[(dm_table["regime"] == regime) & (dm_table["window"] == window)
                               & (dm_table["model_A"] == model_a)
                               & (dm_table["model_B"] == model_b)] if len(dm_table) else pd.DataFrame()
            sig = ordered[ordered["significant_holm"] == True] if len(ordered) else ordered
            robust = holm(np.asarray(per_stock_lag, dtype=float)) if per_stock_lag else np.array([])
            rows.append({
                "regime": regime, "window": int(window),
                "model_a": model_a, "model_b": model_b,
                "stocks": int(panel.shape[1]), "dates": int(t_obs),
                "paired_obs": int(np.isfinite(panel.to_numpy(float)).sum()),
                "mean_d_pct2": pooled["mean_d"], "se": pooled["se"],
                "ci_low": pooled["ci_low"], "ci_high": pooled["ci_high"],
                "pooled_stat": pooled["stat"], "pooled_p": pooled["p"],
                "pooled_lag": pooled["lag"],
                "mean_z_paper": float(ordered["z_paper"].mean()) if len(ordered) else float("nan"),
                "n_tests": int(len(ordered)),
                "sig_a_worse": int((sig["z_paper"] > 0).sum()) if len(sig) else 0,
                "sig_a_better": int((sig["z_paper"] < 0).sum()) if len(sig) else 0,
                "sig_lag_auto": int(np.nansum(robust < 0.05)) if robust.size else 0,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["pooled_p_holm"] = np.nan
    for _, group in frame.groupby(["regime", "window"]):
        frame.loc[group.index, "pooled_p_holm"] = holm(group["pooled_p"].to_numpy())
    return frame


def compute(run_dir: Path, write: bool = True) -> dict:
    """All the R3.M4 numbers of a run; also written to data/r3m4/ of the run."""
    run_dir = Path(run_dir)
    frames = load_predictions(run_dir)
    if not frames:
        return {}
    dm_path = run_dir / "data" / "dm_per_stock.csv"
    dm_table = pd.read_csv(dm_path) if dm_path.is_file() else pd.DataFrame()
    models = _models(frames)
    example_out = worked_example(frames, dm_table, models)
    example, paired = example_out if example_out else ({}, pd.DataFrame())
    pooled = pooled_tests(frames, dm_table, models)

    per_stock_total = int(len(dm_table)) if len(dm_table) else 0
    unordered = pooled[["n_tests", "sig_a_worse", "sig_a_better", "sig_lag_auto"]].sum() \
        if len(pooled) else pd.Series(dtype=float)
    out = {
        "models": models,
        "example": example,
        "pooled": json.loads(pooled.to_json(orient="records")) if len(pooled) else [],
        "per_stock": {
            "tests_unordered": int(unordered.get("n_tests", 0)),
            "significant_lag0": int(unordered.get("sig_a_worse", 0) + unordered.get("sig_a_better", 0)),
            "significant_lag_auto": int(unordered.get("sig_lag_auto", 0)),
            "tests_in_table": per_stock_total,
        },
        "lags": {f"T{n}": auto_lag(n) for n in sorted({len(f) for f in frames.values()})},
    }
    if write:
        folder = run_dir / "data" / "r3m4"
        folder.mkdir(parents=True, exist_ok=True)
        if len(pooled):
            pooled.to_csv(folder / "pooled_tests.csv", index=False)
        if len(paired):
            paired.to_csv(folder / "worked_example_paired_losses.csv", index=False)
    return out
