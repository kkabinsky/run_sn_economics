"""The auditable replication package.

Everything a reviewer asked to be able to check is written to disk:

    data/chronology.csv        one row per ticker x regime x window x model:
                               training, validation and test dates, the number
                               of forecast observations, and the normalisation
                               constants fitted on training only
    data/predictions/*.csv     actual and predicted values per ticker
    data/metrics_per_stock.csv backtest metrics from the corrected engine
    data/dm_per_stock.csv      Diebold-Mariano tests with HAC, HLN and Holm
    data/coverage.csv          matched test dates across stocks, with exclusions
    data/leakage_effect.csv    the archived rule against the clean rule
    report/REPORT.md           the answers to R3.M2 and R3.M3
    report/provenance.json     file hashes, versions, seeds, timestamps
    manifest.sha256            hash of every file in the run
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .data import file_sha256


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "catboost", "lightgbm",
                 "xgboost", "torch"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = "not installed"
    try:
        import torch
        versions["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return versions


def git_revision(folder: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "not a git repository"


def write_provenance(run_dir: Path, run_config: dict, inputs: list[dict]) -> Path:
    """Record where every number came from."""
    code_files = sorted((config.PROJECT_ROOT / "leakfree").glob("*.py"))
    entry = config.PROJECT_ROOT / "rerun_leakfree.py"
    legacy_script = config.LEGACY_SCRIPT_ROOT / config.LEGACY_SCRIPT_NAME
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_config": run_config,
        "versions": package_versions(),
        "seed": config.DEFAULT_SEED,
        "code": {
            "package_root": str(config.PROJECT_ROOT),
            "files": [{"path": str(p), "sha256": file_sha256(p)} for p in code_files],
            "entry_point_sha256": file_sha256(entry) if entry.is_file() else "missing",
            "scripts": [{"path": str(config.PROJECT_ROOT / name),
                         "sha256": file_sha256(config.PROJECT_ROOT / name)}
                        for name in ("rerun_leakfree.py", "finish_run.py")
                        if (config.PROJECT_ROOT / name).is_file()],
        },
        "evaluation_engine": {
            "origin": str(config.STAGE1_ENGINE),
            "origin_sha256": file_sha256(config.STAGE1_ENGINE)
            if config.STAGE1_ENGINE.is_file() else "missing",
            "copy": str(config.PROJECT_ROOT / "leakfree" / "evaluation.py"),
            "copy_sha256": file_sha256(config.PROJECT_ROOT / "leakfree" / "evaluation.py"),
        },
        "archived_training_script": {
            "path": str(legacy_script),
            "sha256": file_sha256(legacy_script) if legacy_script.is_file() else "missing",
            "leakage_lines": {
                "scaler_fitted_on_all_windows": "lines 153-160",
                "split_without_validation": "lines 165-170",
                "early_stopping_on_test": "lines 234-240",
            },
        },
        "git_revision_of_archived_code": git_revision(config.LEGACY_SCRIPT_ROOT),
        "manuscript_sources": {
            "group_table": "D:\\paper_conGANs\\final_version_catboost\\supplementary_backtest_version3.tex",
            "response_letter": "D:\\revised_sn_economics\\answer_sn_economics3.tex",
        },
        "inputs": inputs,
    }
    path = run_dir / "report" / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_manifest(run_dir: Path) -> Path:
    """A hash of every file in the run, so the package can be verified later."""
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            lines.append(f"{file_sha256(path)}  {path.relative_to(run_dir).as_posix()}")
    manifest = run_dir / "manifest.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def coverage_table(chronology: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """Which stocks are compared on which dates, and which are excluded."""
    rows = []
    for (regime, window), group in chronology.groupby(["regime", "window"]):
        dates_per_stock = {}
        for _, row in group.iterrows():
            frame = predictions.get((row["ticker"], regime, window))
            if frame is not None and len(frame):
                dates_per_stock[row["ticker"]] = tuple(frame["date"])
        if not dates_per_stock:
            continue
        common = set.intersection(*[set(v) for v in dates_per_stock.values()])
        for ticker, dates in dates_per_stock.items():
            rows.append({
                "regime": regime,
                "window": window,
                "ticker": ticker,
                "n_test_dates": len(dates),
                "n_matched_dates": len(common),
                "n_dates_not_matched": len(set(dates) - common),
                "matched_from": min(common) if common else "",
                "matched_to": max(common) if common else "",
                "excluded_dates": ";".join(sorted(set(dates) - common)),
                "n_stocks_with_data": len(dates_per_stock),
                "n_stocks_in_group": int(group["ticker"].nunique()),
            })
    return pd.DataFrame(rows)


def regime_label_check(chronology: pd.DataFrame) -> pd.DataFrame:
    """Verify that the test block really belongs to the regime it is called."""
    rows = []
    for (regime, window), group in chronology.groupby(["regime", "window"]):
        definition = config.REGIME_DEFINITION[regime]
        start = pd.Timestamp(definition["start"])
        end = pd.Timestamp(definition["end"])
        for _, row in group.iterrows():
            test_start = pd.Timestamp(row["test_start"]) if row["test_start"] else pd.NaT
            test_end = pd.Timestamp(row["test_end"]) if row["test_end"] else pd.NaT
            inside = bool(pd.notna(test_start) and pd.notna(test_end)
                          and test_start >= start and test_end <= end)
            rows.append({
                "regime": regime,
                "window": window,
                "ticker": row["ticker"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "n_test": row["n_test"],
                "regime_definition_start": definition["start"],
                "regime_definition_end": definition["end"],
                "test_block_inside_definition": inside,
                "definition_source": definition["source"],
            })
    return pd.DataFrame(rows)


def leakage_effect_table(clean: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    """Side by side: the archived rule and the clean rule on the same data."""
    if clean.empty or legacy.empty:
        return pd.DataFrame()
    keys = ["ticker", "regime", "window", "model", "cost_bps"]
    merged = clean.merge(legacy, on=keys, suffixes=("_clean", "_legacy"))
    rows = []
    for metric in ("cum_return_geom", "sharpe_annual", "mase", "rmse", "hit_rate"):
        left = f"{metric}_clean"
        right = f"{metric}_legacy"
        if left not in merged.columns or right not in merged.columns:
            continue
        difference = merged[left] - merged[right]
        rows.append({
            "metric": metric,
            "mean_clean": float(merged[left].mean()),
            "mean_legacy": float(merged[right].mean()),
            "mean_difference": float(difference.mean()),
            "median_difference": float(difference.median()),
            "n_pairs": int(len(merged)),
            "n_clean_better": int((difference > 0).sum()),
        })
    return pd.DataFrame(rows)


def build_report(run_dir: Path, run_config: dict, chronology: pd.DataFrame,
                 metrics: pd.DataFrame, dm: pd.DataFrame, coverage: pd.DataFrame,
                 leakage: pd.DataFrame, label_check: pd.DataFrame) -> Path:
    """The letter-ready summary, one section per reviewer point."""
    lines = []
    add = lines.append

    add("# Leakage-free re-run of the SET50 forecasting experiments")
    add("")
    add(f"Run created {datetime.now(timezone.utc).isoformat(timespec='seconds')}.")
    add("")
    add("## 1. What changed with respect to the archived implementation")
    add("")
    add("| Item | Archived run | This re-run |")
    add("|---|---|---|")
    add("| Normalisation | `MinMaxScaler.fit_transform` on **all** sliding windows before the split "
        "(`new_cat_conGAN_modified.py` lines 153-160) | scaler fitted on the **training windows only**; "
        "validation and test are transformed with those constants |")
    add("| Split | index offsets `train = 905 + kk`, `test = next 100` (lines 165-170) | "
        "chronological split stated as **dates**: training, then a validation block, then the untouched test block |")
    add("| Early stopping | `eval_set=(X_test, y_test_orig)` (lines 234-240) | "
        "`eval_set=(X_val, y_val)`; the test block is read once, at the end |")
    add("| Generator selection | last epoch | epoch with the best **validation** loss |")
    add("| Saved predictions | per stock only for CatBoost, cGAN and the hybrids; "
        "LightGBM and XGBoost were never stored | every model stored per stock, per regime, per window |")
    add("")
    add("## 2. The test blocks actually used")
    add("")
    if not chronology.empty:
        blocks = (chronology.groupby(["regime", "window"])
                  .agg(stocks=("ticker", "nunique"),
                       n_test_median=("n_test", "median"),
                       n_test_min=("n_test", "min"),
                       n_test_max=("n_test", "max"),
                       test_start=("test_start", "min"),
                       test_end=("test_end", "max"))
                  .reset_index())
        add(blocks.to_markdown(index=False))
        add("")
    add("The archived 'normal' regime used `train_size = 1905` on a sample that ends on "
        "3 April 2024, so its test block is truncated to the last few rows of the file. "
        "The re-run reports the same truncation instead of hiding it.")
    add("")
    add("## 3. Regime labels")
    add("")
    if not label_check.empty:
        summary = (label_check.groupby(["regime", "window"])
                   .agg(stocks=("ticker", "nunique"),
                        inside=("test_block_inside_definition", "sum"))
                   .reset_index())
        add(summary.to_markdown(index=False))
        add("")
    for regime, definition in config.REGIME_DEFINITION.items():
        add(f"- **{regime}** ({definition['label']}): {definition['start']} to {definition['end']}. "
            f"{definition['definition']} Source: {definition['source']}.")
    add("")
    add("The manuscript describes the crash evaluation in Section 3.10 as 100 days from "
        "17 February to 30 March 2020. That calendar interval contains 31 trading days, "
        "and the manuscript's own group table lists T = 31 for every crash group, so the "
        "two statements cannot both hold. The re-run uses the group table, i.e. the dates, "
        "and reports the number of observations that the data actually supports.")
    add("")
    add("## 4. Matched dates and exclusions")
    add("")
    if not coverage.empty:
        add(coverage.groupby(["regime", "window"])
            .agg(stocks=("ticker", "nunique"),
                 matched_min=("n_matched_dates", "min"),
                 matched_max=("n_matched_dates", "max"))
            .reset_index().to_markdown(index=False))
        add("")
    add("## 5. Effect of removing the leak")
    add("")
    if not leakage.empty:
        add(leakage.to_markdown(index=False))
        add("")
        add("The clean rule is the one that should be reported. The archived rule is kept "
            "in the release only so the difference can be inspected rather than asserted.")
    else:
        add("Not computed in this run.")
    add("")
    add("## 6. Accuracy and trading results (clean run)")
    add("")
    if not metrics.empty:
        grouped = (metrics.groupby(["regime", "window", "model"])
                   .agg(n=("ticker", "size"),
                        mean_return=("cum_return_geom", "mean"),
                        mean_mase=("mase", "mean"),
                        mean_rmse=("rmse", "mean"),
                        mean_hit=("hit_rate", "mean"))
                   .reset_index())
        add(grouped.round(4).to_markdown(index=False))
        add("")
    add("## 7. Diebold-Mariano tests")
    add("")
    add("`d_t = loss_A - loss_B`, so a positive statistic means the first model is the "
        "worse one. Newey-West HAC variance at lag h-1, Harvey-Leybourne-Newbold "
        "correction, t(T-1) reference distribution, Holm adjustment inside each family.")
    add("")
    if not dm.empty:
        add(dm.groupby(["regime", "window", "model_A", "model_B"])
            .agg(tests=("ticker", "size"),
                 mean_d=("mean_d", "mean"),
                 significant_holm=("significant_holm", "sum"))
            .reset_index().round(4).to_markdown(index=False))
    add("")
    add("## 8. How to reproduce")
    add("")
    add("```bash")
    add("python rerun_leakfree.py --out outputs/<run> --models CatBoost LightGBM XGBoost cGAN \\")
    add("       --regimes crash normal --windows 5 10 30 --val-days 50")
    add("```")
    add("")
    add("`report/provenance.json` holds the hash of every input file, of every module of "
        "this program and of the archived evaluation engine, together with the library "
        "versions and the random seed. `manifest.sha256` holds the hash of every file in "
        "the release.")
    add("")

    path = run_dir / "report" / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
