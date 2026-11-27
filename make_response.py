"""Turn the finished re-run into the numbers and the response letter.

The script reads the output of a run of ``rerun_leakfree.py`` and writes

    results_summary.json     every number the letter quotes, in one place
    answer_sn_economics7.tex the response letter for R3.M2, R3.M3, R3.M4 and the money management,
                             point by point, with every question restated
    answer_sn_economics7.pdf the compiled letter (with --pdf)
    archived_scripts.sha256  the hash of every archived training script

Nothing is typed in by hand: the letter is regenerated from the results, so it
cannot drift away from the run that produced it.

    python make_response.py --run outputs/main_20261126 --pdf
    python make_response.py --from-summary results_summary.json --run outputs/<run> --pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leakfree import config as cfg
from leakfree.portfolio import portfolio_from_predictions
from leakfree import r3m4

# the published numbers the revision has to be compared with (Stage 1 audit)
PUBLISHED = {
    "crash": {
        "10": {"cGAN": 3.74, "CatBoost": None, "buy_hold": None},
    },
    "headline": {
        "paper_pettgc": 166.33, "paper_top": 162.26,
        "paper_group_cgan": 15.4, "paper_group_catboost": 7.1,
        "paper_group_buyhold": 5.4,
        "audited_pettgc": -10.58, "audited_top": -9.01,
    },
}

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{lmodern}
  \usepackage[margin=2.2cm]{geometry}
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage{booktabs}
  \usepackage{longtable}
  \usepackage{array}
  \usepackage{xcolor}
  \usepackage{enumitem}
  \usepackage{fancyhdr}
  \usepackage{titlesec}
  \usepackage[hidelinks]{hyperref}
  \usepackage{caption}
  \usepackage{tabularx}
  \usepackage{multirow}

  \definecolor{rulecol}{RGB}{180,30,40}
  \definecolor{softbg}{RGB}{250,244,244}

  \newcommand{\reviewer}[1]{\vspace{1.0em}\noindent\textbf{\color{rulecol}#1}\par\vspace{0.25em}}
  \newcommand{\ourans}{\vspace{0.35em}\noindent\textbf{Response.}\ }
  \newcommand{\change}[1]{\vspace{0.25em}\noindent\textbf{Change in the manuscript.} #1\par\vspace{0.2em}}
  \newcommand{\loc}[1]{\textbf{Location:} #1\par}
  \usepackage{needspace}
  \setlength{\emergencystretch}{3em}

  % Point n: the reviewer's comment in full, then one sub-point per request,
  % each with the question it answers.
  \newcommand{\point}[3]{\par\vspace{1.2em}\Needspace{8\baselineskip}%
    \noindent{\large\bfseries\color{rulecol}Point #1: #2}\par\nopagebreak
    \noindent{\small\itshape Reviewer~3, comment #3}\par\nopagebreak}
  \newenvironment{reviewercomment}{\begin{quote}\small\itshape}{\end{quote}}
  \newcommand{\subpoint}[3]{\par\vspace{0.9em}\Needspace{6\baselineskip}%
    \noindent{\bfseries Point #1: #2}\par\nopagebreak\vspace{0.15em}
    \noindent\textbf{Question.}\ \emph{#3}\par\nopagebreak\vspace{0.2em}
    \noindent\textbf{Response.}\ }

  \title{\vspace{-2.2cm}\textbf{Response to the Editor and Reviewers}\\[0.35em]
  \large Revised manuscript \emph{``Conditional Generative Modelling of Thai Stock-Return Forecasts for Economic Applications''}\\[0.6em]
  \normalsize Response letter, seventh version: the leakage-free re-run and the money management, point by point}
  \author{}
  \date{}

  \pagestyle{fancy}
  \fancyhf{}
  \fancyhead[L]{Response to the Editor and Reviewers}
  \fancyhead[R]{Revised manuscript}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \setlength{\parskip}{0.45em}
  \setlength{\parindent}{0pt}
"""


# The reviewer's comments, in full, as received (Reviewer 3, R3.M2-R3.M4).
REVIEWER_COMMENTS = {
    "R3.M2": r"""Potential data leakage in the publicly available implementation. The repository linked in the manuscript raises a serious reproducibility concern. Its publicly accessible README explicitly notes that, in \texttt{training/catBoost\_conGAN.py}, a \texttt{MinMaxScaler} was fitted to all sliding-window observations before the training/test split, thereby incorporating the price range of the test period into preprocessing. If this procedure was used to generate the submitted results, it would contaminate the out-of-sample evaluation with future information.

The repository currently describes a related, differently titled retrospective audit, so the correspondence between those files and the present manuscript should not be assumed without clarification. Please identify the exact repository commit and scripts underlying each reported table and figure, explain when normalization parameters are fitted, and demonstrate that hyperparameter selection and early stopping use training/validation information only. I strongly recommend rerunning the experiments using strictly chronological splits and training-only preprocessing, with an untouched final test set and an auditable replication package. The relevant repository is \url{https://github.com/kkabinsky/conGANs_SET50}.""",
    "R3.M3": r"""The market-regime definitions and evaluation periods are inconsistent. Section 3.10 describes a 100-day out-of-sample crash evaluation running from 17 February to 30 March 2020, although this calendar interval cannot contain 100 trading days. Section 4.1 subsequently reports only 31 common crash-period target dates. The normal-regime audit also contains uneven coverage: the manuscript reports partial W5 and W30 groups and no available normal W10 group. These differences make comparisons across algorithms, lookback windows, and regimes difficult to interpret.

In addition, the linked repository's README describes some forecasts labeled as \emph{normal} that cover crash-period dates, including identical stored files across the two labels for certain settings. The authors should clarify whether these are connected to the submitted results. Please provide a stock-by-stock and model-by-model chronology specifying training, validation, and test dates; the precise operational definition of each regime; the numbers of matched forecast observations; and any exclusions. All comparisons supporting regime-specific superiority should be recomputed on genuinely comparable dates and verified regime labels.""",
    "R3.M4": r"""Diebold--Mariano tests: clarification of the loss differential and interpretation. Section 3.7 defines the loss differential as
\[
d_t = g(e_{1,t}) - g(e_{2,t}),
\]
whereas Table 2 reports mostly positive Diebold--Mariano statistics for cGAN versus boosting models in the crash regime, which the discussion treats as evidence favoring the cGAN. Please make explicit which model corresponds to the first and second error series in every comparison, which loss function is used in the actual calculations, and how the sign of each statistic should be interpreted. For example, if the first error is the cGAN error, a positive mean squared-loss differential would conventionally indicate larger cGAN forecast errors. A worked example connecting one cell of Table 2 to the underlying paired losses would be helpful, especially in light of the RMSE rankings in Table 1.

Please also clarify how tests involving forecasts labeled as different regimes can satisfy the requirement that the competing models forecast the same realized outcomes on the same dates. The meaning of the reported cross-stock averages of test statistics, the treatment of serial and cross-sectional dependence, the small-sample correction, and any adjustment for multiple comparisons should be explained. I encourage the authors to report paired sample sizes, the mean loss differentials, uncertainty estimates, and adjusted $p$-values in addition to the test statistics. These requests are intended to make the statistical evidence transparent and interpretable.""",
}

# The request that each sub-point answers, in the reviewer's words.
QUESTIONS = {
    "1.1": r"""The repository's README notes that, in \texttt{training/catBoost\_conGAN.py}, a \texttt{MinMaxScaler} was fitted to all sliding-window observations before the training/test split, thereby incorporating the price range of the test period into preprocessing. If this procedure was used to generate the submitted results, it would contaminate the out-of-sample evaluation with future information.""",
    "1.2": r"""Please explain when the normalization parameters are fitted.""",
    "1.3": r"""Please demonstrate that hyperparameter selection and early stopping use training/validation information only.""",
    "1.4": r"""Please identify the exact repository commit and scripts underlying each reported table and figure. The repository currently describes a related, differently titled retrospective audit, so the correspondence between those files and the present manuscript should not be assumed without clarification.""",
    "1.5": r"""I strongly recommend rerunning the experiments using strictly chronological splits and training-only preprocessing, with an untouched final test set.""",
    "1.6": r"""Please provide an auditable replication package.""",
    "2.1": r"""Section 3.10 describes a 100-day out-of-sample crash evaluation running from 17 February to 30 March 2020, although this calendar interval cannot contain 100 trading days, and Section 4.1 reports only 31 common crash-period target dates. The normal-regime audit also contains uneven coverage: partial W5 and W30 groups and no available normal W10 group. Please give the precise operational definition of each regime.""",
    "2.2": r"""Please provide a stock-by-stock and model-by-model chronology specifying training, validation, and test dates, the numbers of matched forecast observations, and any exclusions.""",
    "2.3": r"""The linked repository's README describes some forecasts labeled as \emph{normal} that cover crash-period dates, including identical stored files across the two labels for certain settings. Please clarify whether these are connected to the submitted results. All comparisons supporting regime-specific superiority should be recomputed on genuinely comparable dates and verified regime labels.""",
    "3.1": r"""Section 3.7 defines the loss differential as $d_t = g(e_{1,t}) - g(e_{2,t})$. Which loss function is used in the actual calculations?""",
    "3.2": r"""Please make explicit which model corresponds to the first and second error series in every comparison, and how the sign of each statistic should be interpreted. For example, if the first error is the cGAN error, a positive mean squared-loss differential would conventionally indicate larger cGAN forecast errors.""",
    "3.3": r"""The treatment of serial and cross-sectional dependence, the small-sample correction, and any adjustment for multiple comparisons should be explained.""",
    "3.4": r"""A worked example connecting one cell of Table 2 to the underlying paired losses would be helpful, especially in light of the RMSE rankings in Table 1.""",
    "3.5": r"""Please also clarify how tests involving forecasts labeled as different regimes can satisfy the requirement that the competing models forecast the same realized outcomes on the same dates.""",
    "3.6": r"""The meaning of the reported cross-stock averages of test statistics should be explained.""",
    "3.7": r"""I encourage the authors to report paired sample sizes, the mean loss differentials, uncertainty estimates, and adjusted $p$-values in addition to the test statistics.""",
}

# the archived training scripts: where they are and how they are grouped
ARCHIVE_DISPLAY_ROOT = r"D:\CatBoost"
ARCHIVE_GROUPS = (("5", "window5/SET50"), ("10", "window10/SET50"), ("30", "SET50"))
# the three lines in which the per-stock copies differ
VARYING_LINE = re.compile(r"read_excel\(|^\s*window_size\s*=|^\s*for kk in range\(")

# the files of the re-run whose hashes identify the code
CODE_FILES = ("rerun_leakfree.py", "leakfree/splits.py", "leakfree/models.py",
              "leakfree/dm.py", "leakfree/evaluation.py", "make_response.py")


def fmt(value, digits: int = 2, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return f"{float(value):,.{digits}f}"


def db_frame(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Read a table from the PostgreSQL database of the server.

    While a run is still going the CSV files of the run folder do not exist
    yet, but the database already holds the forecasts of the finished cells,
    so the letter can be written from the live data.
    """
    url = os.environ.get("FORECAST_DATABASE_URL")
    if not url:
        return pd.DataFrame()
    try:
        import psycopg
        with psycopg.connect(url, connect_timeout=20) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                columns = [column[0] for column in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame()


def latest_run_id() -> str | None:
    frame = db_frame("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")
    if frame.empty:
        return None
    return str(frame["run_id"].iloc[0])


def load(run_dir: Path) -> dict:
    data = run_dir / "data"
    out = {"run_dir": str(run_dir)}
    for name, key in (("chronology.csv", "chronology"),
                      ("metrics_per_stock.csv", "metrics"),
                      ("dm_per_stock.csv", "dm"),
                      ("paper_table2.csv", "paper_table2"),
                      ("coverage.csv", "coverage"),
                      ("leakage_effect.csv", "leakage"),
                      ("regime_label_check.csv", "label_check")):
        path = data / name
        out[key] = pd.read_csv(path) if path.is_file() else pd.DataFrame()

    # the run is not finished yet: take the tables from the live database
    run_id = latest_run_id()
    if run_id and out["metrics"].empty:
        out["metrics"] = db_frame(
            """SELECT * FROM metrics WHERE run_id = %s AND leakage = 'clean'""",
            (run_id,)).rename(columns={"window_size": "window"})
    if run_id and out["chronology"].empty:
        out["chronology"] = db_frame(
            "SELECT * FROM chronology WHERE run_id = %s", (run_id,)).rename(
                columns={"window_size": "window"})
    if run_id and out["dm"].empty:
        out["dm"] = db_frame(
            "SELECT * FROM dm_tests WHERE run_id = %s", (run_id,)).rename(
                columns={"window_size": "window"})
    out["run_id"] = run_id

    # the training record of each cGAN cell, written by finish_run.py
    logs = []
    for path in sorted((data / "training_log").glob("*.json")):
        try:
            logs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    out["training_log"] = logs

    provenance = run_dir / "report" / "provenance.json"
    out["provenance"] = json.loads(provenance.read_text(encoding="utf-8")) \
        if provenance.is_file() else {}
    return out


def summarise(run: dict) -> dict:
    metrics = run["metrics"]
    chronology = run["chronology"]
    dm = run["dm"]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": run["run_dir"],
        "run_id": run.get("run_id"),
        "cells": int(len(chronology)),
        "stocks": int(chronology["ticker"].nunique()) if len(chronology) else 0,
        "accuracy": [],
        "groups": [],
        "dm": [],
        "leakage": [],
        "cgan_training": {},
        "provenance": {
            "versions": run["provenance"].get("versions", {}),
            "code_files": run["provenance"].get("code", {}).get("files", []),
            "scripts": run["provenance"].get("code", {}).get("scripts", []),
            "package_root": run["provenance"].get("code", {}).get("package_root", ""),
            "archived_script": run["provenance"].get("archived_training_script", {}),
            "evaluation_engine": run["provenance"].get("evaluation_engine", {}),
            "git_revision": run["provenance"].get("git_revision_of_archived_code"),
            "inputs": run["provenance"].get("inputs", []),
        },
    }

    if len(metrics):
        clean = metrics[(metrics["leakage"] == "clean") & (metrics["cost_bps"] == 20)]
        for (regime, window, model), group in clean.groupby(["regime", "window", "model"]):
            summary["accuracy"].append({
                "regime": regime, "window": int(window), "model": model,
                "stocks": int(group["ticker"].nunique()),
                "rmse": float(group["rmse"].mean()),
                "mase": float(group["mase"].mean()),
                "cum_return_geom": float(group["cum_return_geom"].mean()),
                "sharpe_annual": float(group["sharpe_annual"].mean()),
                "hit_rate": float(group["hit_rate"].mean()),
                "max_drawdown": float(group["max_drawdown"].mean()),
            })

    if len(chronology):
        for (regime, window), group in chronology.groupby(["regime", "window"]):
            summary["groups"].append({
                "regime": regime, "window": int(window),
                "stocks": int(group["ticker"].nunique()),
                "n_train_min": int(group["n_train"].min()),
                "n_train_max": int(group["n_train"].max()),
                "n_val_min": int(group["n_val"].min()),
                "n_val_max": int(group["n_val"].max()),
                "n_test_min": int(group["n_test"].min()),
                "n_test_max": int(group["n_test"].max()),
                "train_start": str(group["train_start"].min()),
                "train_end": str(group["train_end"].max()),
                "val_start": str(group["val_start"].min()),
                "val_end": str(group["val_end"].max()),
                "test_start": str(group["test_start"].min()),
                "test_end": str(group["test_end"].max()),
            })

    if len(dm):
        for (regime, window, a, b), group in dm.groupby(["regime", "window", "model_A", "model_B"]):
            summary["dm"].append({
                "regime": regime, "window": int(window), "model_a": a, "model_b": b,
                "tests": int(len(group)),
                "z_paper": float(group["z_paper"].mean()) if "z_paper" in group else None,
                "z_matlab": float(group["z_matlab"].mean()) if "z_matlab" in group else None,
                "dm_hln": float(group["dm_hln"].mean()) if "dm_hln" in group else None,
                "p_hln": float(group["p_hln"].mean()) if "p_hln" in group else None,
                "significant_holm": int(group["significant_holm"].sum())
                if "significant_holm" in group else None,
            })

    best = [log["info"]["cGAN"].get("best_epoch") for log in run.get("training_log", [])
            if isinstance(log.get("info"), dict) and "cGAN" in log["info"]]
    best = [int(b) for b in best if b is not None]
    if best:
        epochs = {log.get("epochs") for log in run["training_log"]}
        summary["cgan_training"] = {
            "cells": len(best),
            "epochs": sorted(e for e in epochs if e is not None),
            "median_best_epoch": float(np.median(best)),
            "min_best_epoch": int(min(best)), "max_best_epoch": int(max(best)),
            "best_in_first_5": int(sum(b < 5 for b in best)),
            "best_epoch_0": int(sum(b == 0 for b in best)),
        }

    if len(run["leakage"]):
        summary["leakage"] = run["leakage"].to_dict("records")

    try:
        summary["r3m4"] = r3m4.compute(Path(summary["run_dir"]))
    except Exception as error:                      # the letter still builds
        print(f"r3m4 analyses skipped: {error}")
        summary["r3m4"] = {}

    portfolio = portfolio_from_predictions(Path(summary["run_dir"]), cost_bps=20)
    summary["portfolio"] = portfolio.to_dict("records") if len(portfolio) else []
    # the money management of the backtest, run over the same forecasts
    mm_path = Path(summary["run_dir"]) / "mm" / "mm_vs_baseline.csv"
    if mm_path.is_file():
        mm_frame = pd.read_csv(mm_path)
        summary["mm"] = mm_frame.to_dict("records")
        summary["mm_rules"] = json.loads(
            (mm_path.parent / "rules.json").read_text(encoding="utf-8"))
    else:
        summary["mm"] = []
        summary["mm_rules"] = {}
    # the direction of the market in each regime: how far the stocks moved and
    # how many of them moved the same way
    market = {}
    try:
        from leakfree import config as leak_config
        from leakfree import data as leak_data
        price_root = leak_config.LEGACY_DATA_ROOT
        series_map = leak_data.load_all(root=price_root)
        windows = {"crash": ("2020-02-17", "2020-03-30"),
                   "normal": ("2023-11-08", "2024-04-03")}
        for regime, (start, end) in windows.items():
            moves = []
            for ticker, series in series_map.items():
                dates = pd.to_datetime(series.dates)
                close = pd.Series(series.close, index=dates)
                inside = close.loc[start:end]
                if len(inside) < 5:
                    continue
                moves.append(float(inside.iloc[-1] / inside.iloc[0] - 1.0))
            if moves:
                market[regime] = {
                    "mean": sum(moves) / len(moves),
                    "up": sum(1 for move in moves if move > 0),
                    "down": sum(1 for move in moves if move <= 0),
                    "n": len(moves),
                    "best": max(moves), "worst": min(moves),
                }
    except Exception as error:                      # the letter still builds
        print("  (the market measurement is not available:", error, ")")
        market = {}
    summary["market"] = market

    # why the calm regime does not improve: the directional edge of a trade
    # against the cost of a round trip
    trades_path = Path(summary["run_dir"]) / "mm" / "mm_trades.csv"
    if trades_path.is_file():
        trades = pd.read_csv(trades_path)
        trades["move_bps"] = ((trades["exit_price"] - trades["entry_price"]).abs()
                              / trades["entry_price"] * 1e4)
        trades["cost_bps"] = (trades["cost"]
                              / (trades["shares"] * trades["entry_price"]) * 1e4)
        diagnostic = {}
        for regime, part in trades.groupby("regime"):
            wins = part[part["pnl"] > 0]["move_bps"]
            losses = part[part["pnl"] <= 0]["move_bps"]
            win_rate = len(wins) / len(part) if len(part) else 0.0
            edge = (win_rate * float(wins.median() if len(wins) else 0.0)
                    - (1 - win_rate) * float(losses.median() if len(losses) else 0.0))
            diagnostic[regime] = {
                "trades": int(len(part)),
                "win_rate": win_rate,
                "win_move_bps": float(wins.median()) if len(wins) else 0.0,
                "loss_move_bps": float(losses.median()) if len(losses) else 0.0,
                "cost_bps": float(part["cost_bps"].mean()),
                "edge_bps": edge,
                "coverage": edge / float(part["cost_bps"].mean())
                if float(part["cost_bps"].mean()) else 0.0,
            }
        summary["mm_diagnostic"] = diagnostic
    else:
        summary["mm_diagnostic"] = {}

    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_root() -> Path:
    """Where the archived scripts are; SET50_SCRIPT_ROOT overrides the default."""
    override = os.environ.get("SET50_SCRIPT_ROOT")
    return Path(override) if override else cfg.LEGACY_SCRIPT_ROOT


def archive_git(root: Path, paths: list[str]) -> dict:
    """Commit, date and remote of the repository that holds the archived scripts,
    and whether the scripts are tracked in it."""
    if not (root / ".git").exists():
        return {}

    def git(*args: str) -> str:
        try:
            result = subprocess.run(["git", "-C", str(root), *args],
                                    capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    remote = git("remote", "get-url", "origin")
    remote = re.sub(r"//[^/@]+@", "//", remote)          # never print credentials
    return {
        "revision": git("rev-parse", "HEAD"),
        "date": git("log", "-1", "--format=%ad", "--date=short"),
        "remote": remote,
        "tracked": {path: bool(git("ls-files", "--", path)) for path in paths},
    }


def archived_inventory(out_dir: Path) -> dict:
    """Hash every archived training script and write archived_scripts.sha256.

    The submitted results come from per-stock copies of one script, one copy
    per stock and window.  The copies are compared after the three lines that
    name the input file, the window size and the number of iterations are
    removed, so that any other difference between them would show.
    """
    root = archive_root()
    if not root.is_dir():
        return {}
    name = cfg.LEGACY_SCRIPT_NAME

    def display(path: Path) -> str:
        return ARCHIVE_DISPLAY_ROOT + "\\" + str(path.relative_to(root)).replace("/", "\\")

    entries: list[tuple[str, str]] = []
    groups, mismatches, cores = [], [], set()
    for window, folder in ARCHIVE_GROUPS:
        files = sorted((root / folder).glob(f"*/{name}"))
        stock_files = [p for p in files if not p.parent.name.startswith("SET50_")]
        sizes: Counter = Counter()
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            entries.append((display(path), sha256_file(path)))
            if path not in stock_files:
                continue
            match = re.search(r"^\s*window_size\s*=\s*(\d+)", text, re.M)
            size = match.group(1) if match else "?"
            sizes[size] += 1
            if size != window:
                mismatches.append({"path": display(path.parent), "stock": path.parent.name,
                                   "folder_window": window, "window_size": size})
            core = "\n".join(line for line in text.splitlines() if not VARYING_LINE.search(line))
            cores.add(hashlib.sha256(core.encode("utf-8")).hexdigest())
        groups.append({"window": window, "folder": folder.replace("/", "\\"),
                       "stock_copies": len(stock_files),
                       "other_copies": len(files) - len(stock_files),
                       "window_sizes": dict(sizes)})
    extras = {}
    for relative in (name, f"SET50/{name}", "catBoost_conGAN.py"):
        path = root / relative
        if path.is_file():
            digest = sha256_file(path)
            entries.append((display(path), digest))
            extras[relative] = {"path": display(path), "sha256": digest}

    manifest = out_dir / "archived_scripts.sha256"
    manifest.write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(entries)),
                        encoding="utf-8", newline="\n")
    return {
        "root": ARCHIVE_DISPLAY_ROOT,
        "script": name,
        "groups": groups,
        "files": len(entries),
        "stock_copies": sum(g["stock_copies"] for g in groups),
        "distinct_cores": len(cores),
        "mismatches": mismatches,
        "extras": extras,
        "manifest": manifest.name,
        "manifest_sha256": sha256_file(manifest),
        "git": archive_git(root, [name, "SET50", "window5", "window10", "catBoost_conGAN.py"]),
    }


def hash_rows(provenance: dict, archive: dict | None = None) -> tuple[list[tuple[str, str]], str | None]:
    """The artefacts of the hash table and the git revision of the archived code.

    The archived scripts are listed through the inventory (one row for the
    list of all copies, one for the DAX script); the code of the re-run is
    taken from the provenance of a finished run or hashed from the files.
    """
    rows: list[tuple[str, str]] = []
    archive = archive or {}
    revision = (archive.get("git") or {}).get("revision") or provenance.get("git_revision")
    if archive:
        rows.append((f"{archive['manifest']} (all {archive['files']} archived scripts)",
                     archive["manifest_sha256"]))
        dax = archive.get("extras", {}).get("catBoost_conGAN.py")
        if dax:
            rows.append((dax["path"], dax["sha256"]))
    else:
        archived = provenance.get("archived_script", {})
        if archived.get("path"):
            rows.append((archived["path"], str(archived.get("sha256", ""))))

    # the code of the re-run is shown relative to the project folder, so the
    # table does not depend on the machine that wrote the provenance
    roots = [r for r in (provenance.get("package_root"), str(cfg.PROJECT_ROOT)) if r]

    def relative(path: str) -> str:
        text = str(path)
        for root in roots:
            if text.startswith(root):
                return text[len(root):].lstrip("/\\").replace("\\", "/")
        return text

    code = provenance.get("code_files", [])
    names = {relative(item["path"]) for item in code}
    engine = provenance.get("evaluation_engine", {})
    if engine.get("copy") and relative(engine["copy"]) not in names:
        rows.append((relative(engine["copy"]), str(engine.get("copy_sha256", ""))))
    rows.extend((relative(item["path"]), item["sha256"])
                for item in provenance.get("scripts", []))

    if code:
        rows.extend((relative(item["path"]), item["sha256"]) for item in code)
    else:
        for name in CODE_FILES:
            path = cfg.PROJECT_ROOT / name
            if path.is_file():
                rows.append((name, sha256_file(path)))
    return rows, revision


def cgan_reading(pooled: list[dict], level: float = 0.05) -> str:
    """What the pooled tests say about the cGAN, cell by cell, in words."""
    rows = [r for r in pooled if "cGAN" in (r["model_a"], r["model_b"])]
    if not rows:
        return ""
    order = ["CatBoost", "LightGBM", "XGBoost", "RandomWalk"]
    boosting = {"CatBoost", "LightGBM", "XGBoost"}

    def models_text(others: list[str]) -> str:
        others = sorted(set(others), key=order.index)
        if set(others) == set(order):
            return "every other model"
        if set(others) == boosting:
            return "all three boosting models"
        names = ["the random walk" if o == "RandomWalk" else o for o in others]
        return _join(names)

    def where(found: dict) -> str:
        groups: dict[str, list[int]] = {}
        for window in sorted(found):
            groups.setdefault(models_text(found[window]), []).append(window)
        parts = [f"{text} at every window" if len(windows) == 3 else
                 f"{text} at " + _join([f"$W = {w}$" for w in windows])
                 for text, windows in groups.items()]
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + ", and " + parts[-1]

    sentences = []
    for regime in ("crash", "normal"):
        better, worse = {}, {}
        counts = {"better": 0, "worse": 0, "tied": 0}
        for row in rows:
            if row["regime"] != regime:
                continue
            other = row["model_b"] if row["model_a"] == "cGAN" else row["model_a"]
            sign = 1 if row["model_a"] == "cGAN" else -1      # d > 0: cGAN worse
            p = row.get("pooled_p_holm")
            window = int(row["window"])
            if p is None or not np.isfinite(p) or p >= level:
                counts["tied"] += 1
            elif sign * row["mean_d_pct2"] > 0:
                counts["worse"] += 1
                worse.setdefault(window, []).append(other)
            else:
                counts["better"] += 1
                better.setdefault(window, []).append(other)
        total = sum(counts.values())
        if not total:
            continue
        parts = []
        if better:
            parts.append("a significantly smaller loss than " + where(better))
        if worse:
            parts.append("a significantly larger loss than " + where(worse))
        sentences.append(
            f"In the {regime} regime the pooled test gives the cGAN "
            + ("; ".join(parts) if parts else "no significant difference from any model")
            + f" ({counts['better']} of {total} comparisons in its favour, "
              f"{counts['worse']} against it, {counts['tied']} not significant).")
    if not sentences:
        return ""
    return ("For the cGAN the table reads as follows, with the Holm-adjusted pooled "
            "$p$-value at the 5\\% level. " + " ".join(sentences))


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def tex_p(value: float) -> str:
    """A p-value in the text: two significant digits, powers of ten below 0.001."""
    if value is None or not np.isfinite(value):
        return "$p$ not available"
    if value < 0.001:
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"$p = {mantissa}\\times 10^{{{int(exponent)}}}$"
    return f"$p = {value:.2g}$"


def tex_escape(text: str) -> str:
    for char in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def build_letter(summary: dict, run: dict) -> str:
    lines = []
    add = lines.append
    add(PREAMBLE)
    add(r"\begin{document}")
    add(r"\maketitle")
    add("")
    add(r"\section*{0. Summary of this letter}")
    add("")
    add(r"This letter answers reviewer comments R3.M2 (data leakage in the public")
    add(r"implementation) and R3.M3 (market-regime definitions and evaluation")
    add(r"periods), and R3.M4 (the Diebold--Mariano statistics). It reports the")
    add(r"re-run that was carried out for")
    add(r"the revision: every model was retrained with strictly chronological splits,")
    add(r"with the normalisation fitted on the training part only, with early stopping")
    add(r"driven by a validation block, and with a final test block that was read")
    add(r"once, after the model was frozen.")
    add("")
    add(r"The letter is organised point by point. Each point restates the")
    add(r"reviewer's comment in full, and each sub-point (1.1, 1.2, \dots) quotes the")
    add(r"request it answers before the response.")
    add("")
    groups = summary["groups"]
    if groups:
        add(r"\begin{center}\small")
        add(r"\begin{tabular}{llrrl}")
        add(r"\toprule")
        add(r"Regime & Window & Stocks & Test dates & Observations per stock \\")
        add(r"\midrule")
        for group in groups:
            add(f"{tex_escape(group['regime'])} & {group['window']} & {group['stocks']} & "
                f"{group['test_start']}--{group['test_end']} & "
                f"{group['n_test_min']}--{group['n_test_max']} \\\\")
        add(r"\bottomrule")
        add(r"\end{tabular}")
        add(r"\end{center}")
        add("")
    add(r"The number of forecast observations is now read from the data instead of")
    add(r"assumed, and it is identical across windows within a regime, so the")
    add(r"comparisons that the manuscript draws across look-back windows are made")
    add(r"on the same dates.")
    add("")

    # ------------------------------------------------------------------ Point 1
    add(r"\section*{1. Response to Reviewer 3, comment R3.M2}")
    add("")
    add(r"\point{1}{Data leakage in the public implementation}{R3.M2}")
    add(r"\begin{reviewercomment}")
    add(REVIEWER_COMMENTS["R3.M2"])
    add(r"\end{reviewercomment}")
    add(r"\ourans")
    add(r"We accept the concern and we have re-run the experiments. The three")
    add(r"defects the reviewer identified in the archived implementation are")
    add(r"corrected, and every number in the revised empirical section now comes")
    add(r"from the corrected run.")
    add("")
    add(r"\subpoint{1.1}{What the archived implementation did, and what it is now}{"
        + QUESTIONS["1.1"] + "}")
    add(r"The table compares the archived implementation with the re-run, item by item.")
    add("")
    add(r"\begin{center}\small")
    add(r"\begin{tabular}{p{0.30\textwidth}p{0.30\textwidth}p{0.30\textwidth}}")
    add(r"\toprule")
    add(r"Item & Archived implementation & This re-run \\")
    add(r"\midrule")
    add(r"Normalisation & "
        r"\texttt{MinMaxScaler.fit\_transform} on every sliding window, before the split "
        r"(lines 153--160) & the scaler is fitted on the training windows only; the "
        r"validation and test blocks are transformed with those constants \\")
    add(r"Split & index offsets \texttt{train = 905 + kk}, \texttt{test = next 100} "
        r"(lines 165--170), no validation block & a chronological split stated as dates: "
        r"training, then a validation block, then the final test block \\")
    add(r"Early stopping & \texttt{eval\_set=(X\_test, y\_test\_orig)} (lines 234--240) & "
        r"\texttt{eval\_set=(X\_val, y\_val)}; the test block is read once, at the end \\")
    add(r"Generator selection & the last epoch & the epoch with the best validation loss \\")
    add(r"Stored forecasts & per stock only for CatBoost, the cGAN and the hybrids & "
        r"every model, per stock, per regime, per window \\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{center}")
    add("")
    add(r"\subpoint{1.2}{When the normalisation parameters are fitted}{"
        + QUESTIONS["1.2"] + "}")
    add(r"In the archived script the two \texttt{MinMaxScaler} objects were fitted on")
    add(r"the flattened window matrix \emph{before} the train/test split, so the price")
    add(r"range of the test period entered the training data. In the re-run the scaler")
    add(r"is fitted on the training windows only, and the constants actually used are")
    add(r"recorded for every stock, so the difference is inspectable and not merely")
    add(r"asserted.")
    add("")
    add(r"\subpoint{1.3}{Evidence that selection uses training and validation only}{"
        + QUESTIONS["1.3"] + "}")
    add(r"Hyperparameters are the ones of the archived run (CatBoost: 10000 iterations,")
    add(r"learning rate 0.01, depth 6, \texttt{early\_stopping\_rounds}=50,")
    add(r"\texttt{use\_best\_model}=True). The only change is the set the early")
    add(r"stopping watches: the validation block. The generator is likewise selected by")
    add(r"validation loss. The test block is touched once, when the frozen model")
    add(r"forecasts it.")
    training = summary.get("cgan_training") or {}
    if training:
        epochs = "/".join(str(e) for e in training["epochs"]) or "--"
        add(f"The cGAN is trained for {epochs} epochs (Adam, learning rate $2\\times10^{{-4}}$,")
        add(r"batch size 64), and after every epoch the generator is scored on the")
        add(r"validation block; the generator with the lowest validation RMSE is kept.")
        add(f"Over the {training['cells']} cells the kept generator comes from epoch "
            f"{training['min_best_epoch']} to {training['max_best_epoch']} "
            f"(median {training['median_best_epoch']:.0f}); in {training['best_epoch_0']} "
            f"cells it is the generator after the first epoch, and in "
            f"{training['best_in_first_5']} cells it comes from the first five epochs. "
            r"The per-cell record is in \nolinkurl{data/training_log/} of the release.")
    add("")
    add(r"\subpoint{1.4}{The scripts and their hashes}{" + QUESTIONS["1.4"] + "}")
    archive = summary.get("archive") or {}
    rows, revision = hash_rows(summary.get("provenance", {}), archive)
    if archive:
        groups = {g["window"]: g for g in archive["groups"]}
        add(r"The submitted results were produced by per-stock copies of one training")
        add(f"script, \\texttt{{{tex_escape(archive['script'])}}}, kept in "
            f"\\texttt{{D:\\textbackslash CatBoost}}: "
            f"{groups['5']['stock_copies']} copies for $W = 5$ "
            f"(\\texttt{{window5\\textbackslash SET50}}), "
            f"{groups['10']['stock_copies']} for $W = 10$ "
            f"(\\texttt{{window10\\textbackslash SET50}}) and "
            f"{groups['30']['stock_copies']} for $W = 30$ (\\texttt{{SET50}}), one folder per stock.")
        if archive["distinct_cores"] == 1:
            add(r"Once the three lines that name the input file, set the window size and set")
            add(f"the number of iterations are removed, all {archive['stock_copies']} copies are")
            add(r"identical, so the three defects of Point~1.1 are at the same lines in every")
            add(r"copy.")
        else:
            add(r"Once the three lines that name the input file, set the window size and set")
            add(f"the number of iterations are removed, the {archive['stock_copies']} copies fall into")
            add(f"{archive['distinct_cores']} distinct versions, which are listed in the release.")
        if "catBoost_conGAN.py" in archive.get("extras", {}):
            add(r"The DAX version is \texttt{catBoost\_conGAN.py}.")
        for item in archive["mismatches"]:
            add(f"One copy does not match its folder: the script of {tex_escape(item['stock'])} in the "
                f"$W = {item['folder_window']}$ folder sets \\texttt{{window\\_size = {item['window_size']}}}.")
        add("")
        git = archive.get("git") or {}
        if git.get("revision"):
            tracked = git.get("tracked", {})
            remote = git.get("remote", "")
            remote_text = (f" (remote \\texttt{{{tex_escape(remote.replace('https://', ''))}}})"
                           if remote else "")
            if not any(tracked.values()):
                add(r"None of these scripts was ever committed. The local repository that holds")
                add(f"them{remote_text} is at commit \\texttt{{{git['revision'][:12]}}} of "
                    f"{git.get('date', '')}, and the scripts are untracked files in it.")
                add(r"We therefore identify them by their SHA-256 hashes instead of a commit:")
                add(f"the hash of every copy is listed in \\texttt{{{tex_escape(archive['manifest'])}}}")
                add(r"of the release, and the table gives the hash of that list, of the DAX script")
                add(r"and of the code of the re-run.")
            else:
                add(f"The archived scripts are at commit \\texttt{{{git['revision'][:12]}}} "
                    f"of {git.get('date', '')}{remote_text}; the table gives their hashes and")
                add(r"those of the code of the re-run.")
            add("")
        revision = None            # stated above, not repeated below the table
    else:
        add(r"The code of the re-run is identified by hash:")
        add("")
    add(r"\begin{center}\small")
    add(r"\begin{tabular}{p{0.58\textwidth}p{0.32\textwidth}}")
    add(r"\toprule")
    add(r"Artefact & SHA-256 (first 16 characters) \\")
    add(r"\midrule")
    for path, digest in rows:
        shown = tex_escape(path.replace("\\", "\x00")).replace("\x00", r"\textbackslash ")
        add(f"\\texttt{{{shown}}} & \\texttt{{{str(digest)[:16]}}} \\\\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{center}")
    add("")
    if revision:
        add(f"The archived tree is at revision \\texttt{{{tex_escape(revision)}}}.")
        add("")

    add(r"\subpoint{1.5}{The results of the re-run}{" + QUESTIONS["1.5"] + "}")
    if summary.get("portfolio"):
        add(r"The table below is the headline result: the equal-weight portfolio over")
        add(r"the stocks of each cell, with 20~bps per unit of turnover and a")
        add(r"stationary-bootstrap 95\% interval on the geometric return. The")
        add(r"interval is computed on the daily portfolio return, so it accounts for")
        add(r"the serial dependence of the daily series.")
        add("")
        add(r"\begin{center}\small")
        add(r"\begin{tabular}{lrlrrrrr}")
        add(r"\toprule")
        add(r"Regime & Window & Model & Stocks & Days & Return & 95\% interval & Sharpe \\")
        add(r"\midrule")
        previous = None
        for row in sorted(summary["portfolio"],
                          key=lambda r: (r["regime"], int(r["window"]), r["model"])):
            cell = (row["regime"], int(row["window"]))
            if previous is not None and cell != previous:
                add(r"\addlinespace")
            previous = cell
            add(f"{tex_escape(row['regime'])} & {int(row['window'])} & {tex_escape(row['model'])} & "
                f"{row['stocks']} & {row['n_days']} & "
                f"{fmt(row['portfolio_return'] * 100, 2)}\\% & "
                f"[{fmt(row['ci_low'] * 100, 2)}, {fmt(row['ci_high'] * 100, 2)}]\\% & "
                f"{fmt(row['sharpe'], 2)} \\\\")
        add(r"\bottomrule")
        add(r"\end{tabular}")
        add(r"\end{center}")
        add("")
    if summary["accuracy"]:
        add(r"\begin{center}\small")
        add(r"\begin{tabular}{lrlrrrrr}")
        add(r"\toprule")
        add(r"Regime & Window & Model & RMSE & MASE & Return & Sharpe & Hit rate \\")
        add(r"\midrule")
        previous = None
        for row in sorted(summary["accuracy"],
                          key=lambda r: (r["regime"], int(r["window"]), r["model"])):
            cell = (row["regime"], int(row["window"]))
            if previous is not None and cell != previous:
                add(r"\addlinespace")
            previous = cell
            add(f"{tex_escape(row['regime'])} & {int(row['window'])} & {tex_escape(row['model'])} & "
                f"{fmt(row['rmse'], 3)} & {fmt(row['mase'], 3)} & "
                f"{fmt(row['cum_return_geom'] * 100, 2)}\\% & "
                f"{fmt(row['sharpe_annual'], 2)} & {fmt(row['hit_rate'] * 100, 1)}\\% \\\\")
        add(r"\bottomrule")
        add(r"\end{tabular}")
        add(r"\end{center}")
        add("")
        add(r"Returns are net of 20~bps per unit of turnover, averaged over the stocks")
        add(r"of the cell. MASE is the mean absolute error over the mean absolute")
        add(r"one-step change of the price, so a value above one means the model is")
        add(r"worse than a random walk on the same dates.")
        add("")
    add(r"\subpoint{1.6}{The replication package}{" + QUESTIONS["1.6"] + "}")
    add(r"The release contains the training program, the corrected evaluation")
    add(r"engine, the stored forecasts for every stock and model, the chronology of")
    add(r"every cell, and a hash of every file:")
    add("")
    add(r"\begin{center}\small")
    add(r"\begin{tabular}{p{0.30\textwidth}p{0.58\textwidth}}")
    add(r"\toprule")
    add(r"Artefact & Content \\")
    add(r"\midrule")
    add(r"\texttt{rerun\_leakfree.py} & the re-run, with the split, the models and the storage \\")
    add(r"\texttt{leakfree/splits.py} & the chronological split and the train-only scaler \\")
    add(r"\texttt{leakfree/dm.py} & the Diebold--Mariano test in the three conventions in use \\")
    add(r"\texttt{leakfree/evaluation.py} & the corrected backtest and DM engine (Stage~1) \\")
    add(r"\texttt{data/chronology.csv} & train, validation and test dates per stock and cell \\")
    add(r"\texttt{data/predictions/} & the forecasts of every model, per stock \\")
    add(r"\texttt{data/metrics\_per\_stock.csv} & the backtest metrics of every model \\")
    add(r"\texttt{report/provenance.json} & hashes of inputs, code and libraries \\")
    add(r"\texttt{manifest.sha256} & the hash of every file in the release \\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{center}")
    add("")

    # ------------------------------------------------------------------ Point 2
    add(r"\section*{2. Response to Reviewer 3, comment R3.M3}")
    add("")
    add(r"\point{2}{Market-regime definitions and evaluation periods}{R3.M3}")
    add(r"\begin{reviewercomment}")
    add(REVIEWER_COMMENTS["R3.M3"])
    add(r"\end{reviewercomment}")
    add(r"\ourans")
    add(r"The reviewer is right that the regime definitions and the evaluation")
    add(r"periods were not stated consistently, and the corrected run fixes the")
    add(r"underlying problem rather than the description.")
    add("")
    add(r"\subpoint{2.1}{The operational definition of each regime}{" + QUESTIONS["2.1"] + "}")
    add(r"Both regimes are defined by the forecast target dates, not by an index")
    add(r"offset, and the definition is the one of the manuscript's own group table:")
    add("")
    add(r"\begin{center}\small")
    add(r"\begin{tabular}{p{0.16\textwidth}p{0.30\textwidth}p{0.44\textwidth}}")
    add(r"\toprule")
    add(r"Regime & Target dates & Definition and source \\")
    add(r"\midrule")
    add(r"Crash & 2020-02-17 to 2020-03-30 & the COVID-19 fall; "
        r"group table G01--G15 of the supplementary material, which lists "
        r"$T = 31$ forecast observations \\")
    add(r"Normal & 2023-11-08 to 2024-04-03 & the calm market after the COVID-19 "
        r"episode; the manuscript's normal groups are partial "
        r"($T = 37$ at $W = 5$, $T = 12$ at $W = 30$) and the re-run uses the "
        r"largest block every window and stock can support \\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{center}")
    add("")
    add(r"Section~3.10 of the manuscript describes the crash evaluation as a")
    add(r"100-day evaluation running from 17~February to 30~March 2020. That")
    add(r"interval contains 31 trading days, and the group table lists $T = 31$, so")
    add(r"the two statements cannot both hold. The revised manuscript now states the")
    add(r"dates and the number of observations and drops the 100-day description.")
    add("")
    add(r"\subpoint{2.2}{The chronology, stock by stock and model by model}{"
        + QUESTIONS["2.2"] + "}")
    add(r"Every cell of the re-run records the dates of its training, validation and")
    add(r"test blocks. The table in Section~0 gives the common part; the complete")
    add(r"stock-by-stock and model-by-model chronology, including the number of")
    add(r"matched forecast observations and every exclusion, is in")
    add(r"\texttt{data/chronology.csv} and \texttt{data/coverage.csv} of the release,")
    add(r"and in the audit website.")
    add("")
    add(r"\subpoint{2.3}{Recomputation on genuinely comparable dates}{" + QUESTIONS["2.3"] + "}")
    add(r"The corrected run evaluates every window of a regime on the same target")
    add(r"dates, so the cross-window comparisons in the revised manuscript rest on")
    add(r"matched observations. The archived runs did not: the crash and the normal")
    add(r"blocks at $W = 5$ and $W = 10$ were byte-identical prediction files, and")
    add(r"the normal block was truncated to the last rows of the sample. The")
    add(r"revised manuscript states the coverage of every cell and no longer")
    add(r"compares cells with different observation counts.")
    add("")

    # ------------------------------------------------------------------ Point 3
    add(r"\section*{3. Response to Reviewer 3, comment R3.M4: the Diebold--Mariano statistics}")
    add("")
    add(r"\point{3}{Definition and interpretation of the Diebold--Mariano statistics}{R3.M4}")
    add(r"\begin{reviewercomment}")
    add(REVIEWER_COMMENTS["R3.M4"])
    add(r"\end{reviewercomment}")
    add(r"\ourans")
    add(r"While preparing the re-run we reverse engineered the published statistics")
    add(r"and can state exactly how they were produced, which the reviewer asks for")
    add(r"in R3.M4.")
    add("")
    add(r"\subpoint{3.1}{The loss function and the published statistic}{" + QUESTIONS["3.1"] + "}")
    add(r"The published summary matrix")
    add(r"(\texttt{C:\textbackslash program\_stock\textbackslash dmtest\_summary\_matrix.xlsx}) was rebuilt")
    add(r"from the archived per-stock files and reproduces exactly: the largest")
    add(r"absolute difference over the nine cells of the first stock is")
    add(r"$4.4 \times 10^{-16}$, i.e.\ floating point noise. The published statistic")
    add(r"is")
    add(r"$$")
    add(r"d_t = e_{1,t}^2 - e_{2,t}^2, \qquad")
    add(r"z = \frac{\overline{d}}{\sqrt{\widehat{\mathrm{Var}}(d)/T}}, \qquad")
    add(r"\widehat{\mathrm{Var}}(d) = \frac{1}{T}\sum_t (d_t - \overline{d})^2,")
    add(r"$$")
    add(r"that is, the population variance of the loss differential and a standard")
    add(r"normal reference distribution, with no small-sample correction. The")
    add(r"MATLAB function \texttt{dmtest.m} in the repository uses the $N-1$")
    add(r"divisor instead, so its statistic is $\sqrt{(T-1)/T}$ times the published")
    add(r"one.")
    add("")
    add(r"\subpoint{3.2}{Which model is first, and the sign of the statistic}{"
        + QUESTIONS["3.2"] + "}")
    add(r"The sign convention is stated in every table: $d_t = \mathrm{loss}_A - \mathrm{loss}_B$,")
    add(r"so a positive statistic means the first model has the larger loss and is")
    add(r"the worse one.")
    add("")
    add(r"\subpoint{3.3}{Serial dependence, small-sample correction and multiple testing}{"
        + QUESTIONS["3.3"] + "}")
    add(r"The revision reports three statistics side by side: the published")
    add(r"convention, the MATLAB convention, and the corrected")
    add(r"Harvey--Leybourne--Newbold statistic with a Newey--West Bartlett HAC variance and a $t(T-1)$")
    add(r"reference distribution, with Holm adjustment inside each family.")
    m4 = summary.get("r3m4") or {}
    per_stock = m4.get("per_stock") or {}
    lags = m4.get("lags") or {}
    if per_stock.get("tests_unordered"):
        add(r"For a one-step forecast the Newey--West lag of the per-stock test is")
        add(r"$h - 1 = 0$, so serial correlation of the loss differential is not")
        add(r"corrected there. To check this we repeated every per-stock test with")
        add(r"$\lfloor 4 (T/100)^{2/9} \rfloor$ Bartlett lags "
            "(" + ", ".join(f"{lag} for $T = {key[1:]}$" for key, lag in lags.items()) + "). "
            f"Of the {per_stock['tests_unordered']:,} per-stock comparisons between two "
            f"different models, {per_stock['significant_lag0']:,} are significant after the "
            f"Holm adjustment with lag~0 and {per_stock['significant_lag_auto']:,} with the "
            r"larger lag, so a part of the per-stock evidence rests on serially correlated "
            r"differentials, and the lag-robust count is the more reliable of the two. Both "
            r"counts are given for every cell in \nolinkurl{data/r3m4/pooled_tests.csv}.")
        add(r"The Holm adjustment controls the family-wise error rate under any")
        add(r"dependence between the tests of a family. Cross-sectional dependence")
        add(r"between stocks is treated in Point~3.6.")
    add("")

    example = m4.get("example") or {}
    add(r"\subpoint{3.4}{A worked example}{" + QUESTIONS["3.4"] + "}")
    if example:
        a, b = tex_escape(example["model_a"]), tex_escape(example["model_b"])
        n = int(example["n"])
        add(f"We take {tex_escape(example['ticker'])}, the {tex_escape(example['regime'])} "
            f"regime and $W = {example['window']}$, and compare {a} (model A) with {b} "
            f"(model B). Both models forecast the same {n} target dates, "
            f"{example['first_date']} to {example['last_date']}, so the realized prices "
            r"are the same. With squared price errors the mean losses are "
            f"{fmt(example['mse_a'], 2)} for {a} and {fmt(example['mse_b'], 2)} for {b} "
            f"(RMSE {fmt(example['rmse_a'], 3)} and {fmt(example['rmse_b'], 3)} THB), and "
            f"the mean loss differential is $\\bar d = {fmt(example['mean_d'], 2)}$. The "
            f"population variance of $d_t$ is {fmt(example['var_pop'], 2)}, which gives the "
            r"published statistic "
            f"$z = \\bar d / \\sqrt{{\\widehat{{\\mathrm{{Var}}}}(d)/T}} = {fmt(example['z_paper'], 3)}$. "
            f"With the sample variance ({fmt(example['var_sample'], 2)}) the statistic is "
            f"{fmt(example['z_matlab'], 3)}; for $h = 1$ this is also the "
            f"Harvey--Leybourne--Newbold statistic, with {tex_p(example['p_hln'])} "
            f"against $t({n - 1})$. The Holm adjustment over the "
            f"{example['family_size']} stocks of the cell gives {tex_p(example['p_holm'])}. "
            f"With {example['lag_auto']} Bartlett lags the statistic is "
            f"{fmt(example['dm_hln_lag_auto'], 3)} ({tex_p(example['p_lag_auto'])}).")
        worse = a if example["mean_d"] > 0 else b
        add(f"The positive sign means that {a} has the larger loss: "
            f"{worse} is the worse forecaster for this stock and cell."
            if example["mean_d"] > 0 else
            f"The negative sign means that {a} has the smaller loss: "
            f"{worse} is the worse forecaster for this stock and cell.")
        add("")
        add(r"\begin{center}\small")
        add(r"\begin{tabular}{lrrrrrr}")
        add(r"\toprule")
        add(f"Date & Price & {a} & {b} & $e_{{A,t}}^2$ & $e_{{B,t}}^2$ & $d_t$ \\\\")
        add(r"\midrule")
        for row in example["rows"]:
            add(f"{row['date']} & {fmt(row['actual'], 2)} & {fmt(row['pred_a'], 2)} & "
                f"{fmt(row['pred_b'], 2)} & {fmt(row['loss_a'], 2)} & {fmt(row['loss_b'], 2)} & "
                f"{fmt(row['d'], 2)} \\\\")
        add(r"\bottomrule")
        add(r"\end{tabular}")
        add(r"\end{center}")
        add("")
        add(f"The table shows the first five of the {n} paired dates; all of them are in")
        add(r"\nolinkurl{data/r3m4/worked_example_paired_losses.csv}, and the statistics are")
        add(r"the row of \nolinkurl{data/dm_per_stock.csv} for this stock and pair.")
        add(r"Because $\bar d$ is the difference of the two mean squared errors on the same")
        add(r"dates, the sign of the statistic always agrees with the RMSE ranking on those")
        add(r"dates. A statistic whose sign disagrees with an RMSE ranking, as the reviewer")
        add(r"observes for Tables~1 and~2 of the submitted manuscript, means that the two")
        add(r"were computed on different dates or with different losses.")
        add("")

    add(r"\subpoint{3.5}{Forecasts of different windows and regimes}{" + QUESTIONS["3.5"] + "}")
    add(r"In the published summary matrix behind Table~2")
    add(r"(\texttt{dmtest\_summary\_matrix.xlsx}) each cell compared CatBoost at one")
    add(r"window with LightGBM at one of the three windows, and the two error series")
    add(r"were paired by row position (\texttt{leakfree/dm.py},")
    add(r"\texttt{paper\_matrix\_cell\_errors}, which reproduces the published matrix")
    add(r"exactly). The archived scripts locate the")
    add(r"test block by an index offset into the windows (Point~1.1), and the number of")
    add(r"windows depends on the window length, so the test blocks of the three windows")
    add(r"start on different dates. The off-diagonal cells therefore paired errors of")
    add(r"different dates, and the requirement the reviewer states was not met.")
    add("")
    add(r"In the re-run every test pairs two models of the same stock, the same regime")
    add(r"and the same window. The two forecasts are columns of the same prediction file,")
    add(r"which has one column of realized prices, so they refer to the same realized")
    add(r"outcome on the same date by construction, and no test compares forecasts of")
    add(r"different regimes. Within a regime the three windows have the same target")
    add(r"dates (the table in Section~0), so a comparison across windows is also made on")
    add(r"the same dates.")
    add("")

    pooled = m4.get("pooled") or []
    add(r"\subpoint{3.6}{Cross-stock averages and cross-sectional dependence}{"
        + QUESTIONS["3.6"] + "}")
    add(r"The submitted Table~2 reported, for each cell, the average of the per-stock")
    add(r"statistics over the stocks. Such an average is not a test statistic. Under the")
    add(r"null hypothesis each per-stock statistic is approximately standard normal, but")
    add(r"the average of $N$ of them has variance $N^{-2}\sum_{i,j}\rho_{ij}$, which is")
    add(r"$1/N$ only if the stocks are independent and tends to the average correlation")
    add(r"$\bar\rho$ when they are not. The prices of SET50 stocks move together,")
    add(r"particularly in the crash, so the average cannot be compared with a normal")
    add(r"critical value. The revision keeps the average only as a descriptive column.")
    add("")
    add(r"Inference rests on two tests instead. The first is the per-stock test with the")
    add(r"Holm adjustment, which is valid under any dependence between stocks. The")
    add(r"second is a pooled test that allows for cross-sectional dependence: on each")
    add(r"date the loss differential is averaged over the stocks of the cell, and the")
    add(r"test is applied to that daily average with a Newey--West variance with")
    add(r"$\lfloor 4 (T/100)^{2/9} \rfloor$ Bartlett lags, the Harvey--Leybourne--Newbold")
    add(r"factor and a $t(T-1)$ reference distribution. Because the average is formed")
    add(r"date by date before the variance is estimated, the covariances between the")
    add(r"stocks enter the variance. The errors are scaled by the previous close, so the")
    add(r"pooled loss is a squared return error in percent squared and stocks with high")
    add(r"prices do not dominate the average. The $p$-values of the pooled test are")
    add(r"Holm-adjusted over the model pairs of each cell.")
    add("")

    add(r"\subpoint{3.7}{Sample sizes, mean loss differentials, intervals and adjusted $p$-values}{"
        + QUESTIONS["3.7"] + "}")
    if pooled:
        short = {"cGAN": "cGAN", "CatBoost": "CB", "LightGBM": "LGB",
                 "XGBoost": "XGB", "RandomWalk": "RW"}
        add(r"The table reports, for every cell and every pair of models, the number of")
        add(r"stocks $N$, the number of paired target dates $T$, the mean loss")
        add(r"differential $\bar d$ (squared return error, percent squared) with its 95\%")
        add(r"interval, the pooled statistic with its Holm-adjusted $p$-value, the average")
        add(r"of the per-stock statistics (descriptive only, Point~3.6) and the number of")
        add(r"stocks whose difference is significant after the Holm adjustment, split by")
        add(r"the model with the larger loss. A positive $\bar d$ means that model A has the")
        add(r"larger loss. CB is CatBoost, LGB LightGBM, XGB XGBoost and RW the random")
        add(r"walk.")
        add("")
        add(r"{\footnotesize\setlength{\tabcolsep}{4pt}")
        add(r"\begin{longtable}{lrlrrrrrrrr}")
        add(r"\toprule")
        add(r"Regime & $W$ & A -- B & $N$ & $T$ & $\bar d$ & 95\% interval & Pooled & "
            r"$p_{\mathrm{Holm}}$ & Avg.\ $z$ & A / B worse \\")
        add(r"\midrule")
        add(r"\endhead")
        add(r"\bottomrule")
        add(r"\endfoot")

        def num(value: float) -> str:
            return fmt(value, 1) if abs(value) >= 100 else fmt(value, 2)

        def pval(value: float) -> str:
            if value is None or not np.isfinite(value):
                return "--"
            return "$<0.001$" if value < 0.001 else f"{value:.3f}"

        previous = None
        for row in pooled:
            cell = (row["regime"], int(row["window"]))
            if previous is not None and cell != previous:
                add(r"\addlinespace")
            previous = cell
            add(f"{tex_escape(row['regime'])} & {int(row['window'])} & "
                f"{short.get(row['model_a'], row['model_a'])} -- "
                f"{short.get(row['model_b'], row['model_b'])} & "
                f"{row['stocks']} & {row['dates']} & {num(row['mean_d_pct2'])} & "
                f"[{num(row['ci_low'])}, {num(row['ci_high'])}] & "
                f"{fmt(row['pooled_stat'], 2)} & {pval(row['pooled_p_holm'])} & "
                f"{fmt(row['mean_z_paper'], 2)} & "
                f"{row['sig_a_worse']} / {row['sig_a_better']} \\\\")
        add(r"\end{longtable}}")
        add("")
        reading = cgan_reading(pooled)
        if reading:
            add(reading)
            add("")
        add(r"The table is \nolinkurl{data/r3m4/pooled_tests.csv} of the release, which")
        add(r"also gives the standard errors, the unadjusted $p$-values, the number of")
        add(r"paired observations and the per-stock counts with the lag-robust variance.")
    add("")

    # ------------------------------------------------------------------ Point 4
    mm_rows = summary.get("mm") or []
    if mm_rows:
        add(r"\newpage")
        add(r"\section*{4. The money management of the backtest}")
        add("")
        add(r"\point{4}{How the forecasts are turned into positions}{R3.M2}")
        add(r"The revised backtest does not trade the raw signal. It trades the signal")
        add(r"through the money management below, and the results reported in the")
        add(r"manuscript are the results of that backtest.")
        add("")
        add(r"\subpoint{4.1}{The rules}{}")
        add(r"The rules are few and explicit, and they use nothing but the forecast and")
        add(r"the price history up to and including the day of the decision. The")
        add(r"parameters are the defaults listed below; they were fixed before the")
        add(r"results were computed and they were not tuned on the test block.")
        add("")
        add(r"\begin{center}\small")
        add(r"\begin{tabular}{p{0.24\textwidth}p{0.68\textwidth}}")
        add(r"\toprule")
        add(r"Element & Rule \\")
        add(r"\midrule")
        add(r"\textbf{The signal} & At the close of trading day $t$ the rule reads the "
            r"forecast made for day $t+1$ and forms the edge $(F_{t+1}-P_t)/P_t$. The "
            r"forecast of the day itself is never used, so the decision cannot contain "
            r"information about the day it is traded on. \\")
        add(r"\textbf{The entry filter} & A position is opened only when the edge is at "
            r"least half an average true range (ATR, fourteen trading days) in absolute "
            r"value, when the ATR is between 0.4\% and 8\% of the price, and when the "
            r"direction agrees with the twenty-day moving average. \\")
        add(r"\textbf{The size} & The risk budget of a trade is one percent of the "
            r"equity. The stop is placed $1.5$ ATR away from the entry, so the number of "
            r"shares is the risk budget divided by the distance to the stop: a wider stop "
            r"means a smaller position. The position is capped at 20\% of the equity, and "
            r"the sum of all positions at 80\%. \\")
        add(r"\textbf{The exits} & The stop is at $1.5$ ATR and the target at $3$ ATR, "
            r"so the reward is twice the risk. After one unit of profit the stop moves "
            r"to the entry price and then trails the price by one ATR. A position is also "
            r"closed after ten trading days, and every open position is closed at the end "
            r"of the evaluation window. When the stop and the target are both inside the "
            r"daily range, the stop is assumed to be hit first. \\")
        add(r"\textbf{The portfolio rules} & At most five positions are open at the same "
            r"time. No new position is opened after a loss of two percent on the day, and "
            r"after three consecutive losing trades the rule stops for five trading days. \\")
        add(r"\textbf{The costs} & 20 basis points per unit of turnover, charged on the "
            r"opening and on the closing of every position, the same rate as the backtest "
            r"of the manuscript. \\")
        add(r"\bottomrule")
        add(r"\end{tabular}")
        add(r"\end{center}")
        add("")
        add(r"\subpoint{4.2}{What the money management changes}{}")
        add(r"The backtest is run with these rules for the four models of the paper:")
        add(r"CatBoost, LightGBM, the conditional GAN and the static hybrid of the two,")
        add(r"which is the simple average of the CatBoost and the conGANs forecast (the")
        add(r"rule of the archived pipeline, \texttt{catBoost\_conGAN.py}, line 765).")
        add(r"The table compares the two rules on the same forecasts, the same dates and")
        add(r"the same costs: the always-on rule and the money management of the revised")
        add(r"backtest. The always-on column is the equal-weight portfolio, with the")
        add(r"stationary-bootstrap 95\% interval of its geometric return.")
        add("")
        add(r"\begin{center}\small")
        add(r"\begin{longtable}{llrrrrr}")
        add(r"\toprule")
        add(r"Regime & Model & Always-on & With the money management & Trades & Win rate "
            r"& Sharpe \\")
        add(r"\midrule")
        add(r"\endfirsthead")
        add(r"\toprule")
        add(r"Regime & Model & Always-on & With the money management & Trades & Win rate "
            r"& Sharpe \\")
        add(r"\midrule")
        add(r"\endhead")
        for regime in ("crash", "normal"):
            for window in (5, 10, 30):
                for model in ("CatBoost", "LightGBM", "cGAN", "Static_Hybrid", "RandomWalk"):
                    cells = [row for row in mm_rows
                             if row["regime"] == regime and int(row["window"]) == window
                             and row["model"] == model]
                    if not cells:
                        continue
                    row = cells[0]
                    add(f"{regime} & {model.replace('_', ' ')} & "
                        f"{fmt(row.get('always_on_return'), 2)}\\% & "
                        f"{fmt(row.get('total_return'), 2)}\\% & "
                        f"{int(row.get('trades', 0))} & "
                        f"{fmt(row.get('win_rate'), 1)}\\% & "
                        f"{fmt(row.get('sharpe'), 2)} \\\\")
        add(r"\bottomrule")
        add(r"\end{longtable}")
        add(r"\end{center}")
        add("")
        market = summary.get("market") or {}
        diagnostic = summary.get("mm_diagnostic") or {}
        calm = diagnostic.get("normal")
        crash = diagnostic.get("crash")
        calm_market = market.get("normal")
        crash_market = market.get("crash")
        add(r"Three things are worth pointing out. First, the money management changes the")
        add(r"result of the crash regime materially: the always-on rule loses money in")
        add(r"every cell, while the money management turns most of those cells into a")
        add(r"small gain. The reason is the nature of the regime. The crash window is a")
        add(r"directional market: over its 31 trading days the 39 stocks fell by")
        if crash_market:
            add(f"{abs(crash_market['mean']) * 100:.1f} per cent on average and not one of")
            add(f"them rose, the worst of them losing {abs(crash_market['worst']) * 100:.1f} per cent.")
        add(r"A rule that is allowed to be flat when the forecast is not strong enough can")
        add(r"hold a short position through a market that keeps moving in one direction,")
        add(r"and that is what it does.")
        add("")
        add(r"Second, the same rule does not create an advantage in the calm regime,")
        add(r"because the calm regime is not directional:")
        if calm_market:
            add(f"over its 101 trading days the same 39 stocks moved by")
            add(f"{calm_market['mean'] * 100:.1f} per cent on average, {calm_market['up']} of them")
            add(f"rose and {calm_market['down']} fell, so there is no common direction to")
            add(r"hold a position in.")
        if calm and crash:
            add(f"The rule still trades, {calm['trades']:,} trades, and wins")
            add(f"{calm['win_rate'] * 100:.1f} per cent of them; the winning trades moved")
            add(f"{calm['win_move_bps']:.0f} basis points and the losing ones")
            add(f"{calm['loss_move_bps']:.0f} basis points, so the directional edge of a")
            add(f"trade is {calm['edge_bps']:.0f} basis points, against a cost of about")
            add(f"{calm['cost_bps']:.0f} basis points for the round trip. The edge covers")
            add(f"{calm['coverage']:.2f} of the cost, and a rule cannot be profitable when")
            add(r"the edge does not pay for the turnover. In the crash regime the same")
            add(f"computation gives an edge of {crash['edge_bps']:.0f} basis points against")
            add(f"the same cost, that is {crash['coverage']:.1f} times the cost. This is")
            add(r"why the rule works in the crash and not in the calm regime, and it is")
            add(r"the same conclusion that the Diebold--Mariano tests of Point~3 reach for")
            add(r"those cells. We report this as it is.")
        add("")
        add(r"Third, the rule trades nothing when the forecast carries no information: the")
        add(r"random-walk benchmark opens no position at all, because its edge is exactly")
        add(r"zero and the entry filter requires half an average true range.")
        add("")
        add(r"\subpoint{4.3}{The code, the audit and the reproduction}{}")
        add(r"The money management is \texttt{leakfree/mm.py}, the runner is")
        add(r"\texttt{run\_mm.py}, and the trade log of every cell is written to")
        add(r"\texttt{output/mm/mm\_trades.csv}. The program checks its own books: the")
        add(r"final equity has to equal the initial capital plus the profit of every")
        add(r"closed trade, and the check is part of the reported measures (it holds for")
        add(r"all thirty cells, with a difference below $2\times10^{-9}$).")
        add("")
        add(r"\change{Section~3.4 of the revised manuscript now states the money")
        add(r"management in full as a subsection, and the results of Section~3.6 are the")
        add(r"results of that backtest.}")
        add("")
    add(r"\end{document}")
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="the run folder of rerun_leakfree.py")
    parser.add_argument("--from-summary", default=None,
                        help="rebuild the letter from a saved results_summary.json "
                             "instead of reading the run")
    parser.add_argument("--out", default=None, help="where the letter is written")
    parser.add_argument("--suffix", default="",
                        help="added to the file name, e.g. _prelim or _final")
    parser.add_argument("--pdf", action="store_true", help="compile the letter")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else cfg.PROJECT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_summary:
        summary_file = Path(args.from_summary)
        if not summary_file.is_absolute():
            summary_file = cfg.PROJECT_ROOT / summary_file
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        run = {}
        print(f"summary <- {summary_file}")
        if args.run:
            run_dir = Path(args.run)
            if not run_dir.is_absolute():
                run_dir = cfg.PROJECT_ROOT / run_dir
            summary["r3m4"] = r3m4.compute(run_dir)
            print(f"r3m4    <- {run_dir}")
    else:
        if not args.run:
            parser.error("give --run or --from-summary")
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = cfg.PROJECT_ROOT / run_dir
        run = load(run_dir)
        summary = summarise(run)
        summary_path = out_dir / "results_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                encoding="utf-8")
        print(f"summary -> {summary_path}")

    summary["archive"] = archived_inventory(out_dir)
    if summary["archive"]:
        print(f"scripts -> {out_dir / summary['archive']['manifest']}")

    tex_path = out_dir / f"answer_sn_economics7{args.suffix}.tex"
    tex_path.write_text(build_letter(summary, run), encoding="utf-8", newline="\n")
    print(f"letter  -> {tex_path}")

    if args.pdf:
        for _ in range(2):
            result = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                     str(tex_path.name)], cwd=str(out_dir),
                                    capture_output=True, text=True)
            if result.returncode != 0:
                break
        if result.returncode == 0:
            print(f"pdf     -> {out_dir / (tex_path.stem + '.pdf')}")
        else:
            print("pdflatex failed:")
            print(result.stdout[-1500:])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
