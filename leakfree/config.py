"""Paths, tickers, regimes and defaults for the leakage-free re-run.

Everything a reviewer would need to reproduce the run is declared here and
written into the run's ``provenance.json`` at the end.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# where the code and the archived data live
# --------------------------------------------------------------------------- #
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# the archived SET50 price files (one per ticker, one sheet 'closed')
LEGACY_DATA_ROOT = Path(os.environ.get("SET50_DATA_ROOT", r"D:\CatBoost\SET50"))

# the archived training scripts, quoted in the report for provenance
LEGACY_SCRIPT_ROOT = Path(r"D:\CatBoost")
LEGACY_SCRIPT_NAME = "new_cat_conGAN_modified.py"

# the corrected evaluation engine (Stage 1), copied verbatim into this package
STAGE1_ENGINE = Path(r"D:\run_revised_conGANs\revised_core.py")

# where the results of this run are written
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# the 39 SET50 constituents with continuous prices over 2016-2024
TICKERS = [
    "ADVANC", "AOT", "BANPU", "BEM", "BBL", "BDMS", "BH", "BTS", "CBG",
    "CENTEL", "COM7", "CPALL", "CPF", "CPN", "DELTA", "EA", "EGCO", "GLOBAL",
    "GPSC", "HMPRO", "INTUCH", "IVL", "KBANK", "KTC", "KTB", "LH", "MTC",
    "MINT", "PTTGC", "PTT", "PTTEP", "RATCH", "SAWAD", "SCC", "TISCO", "TTB",
    "TOP", "TU", "WHA",
]

# look-back windows reported in the paper (W5 / W10 / W30)
WINDOWS = [5, 10, 30]

# models that are retrained here; RandomWalk and BuyAndHold are benchmarks
TRAINED_MODELS = ["CatBoost", "LightGBM", "XGBoost", "cGAN"]
BENCHMARKS = ["RandomWalk", "BuyAndHold"]

# --------------------------------------------------------------------------- #
# regimes
# --------------------------------------------------------------------------- #
# The archived runs located their test block by an index offset, not by date:
#     crash  : train_size = 905 + kk, test_size = 100   (D:\CatBoost)
#     normal : train_size = 1905 + kk, test_size = 100  (D:\CatBoost_normal)
# The offsets are converted into dates by this program from the data itself, so
# the test block is described by dates in the output, as R3.M3 requires.
#
# The manuscript states in Section 3.10 a "100-day out-of-sample crash
# evaluation running from 17 February to 30 March 2020".  Its own supplementary
# group table (D:\paper_conGANs\final_version_catboost\supplementary_backtest_version3.tex,
# lines 30-45) lists T = 31 for every crash group, and the two reading dates
# cover 31 trading days.  The crash regime below is therefore the manuscript's
# own group definition, and the number of forecast observations is reported
# from the data rather than assumed.
LEGACY_TRAIN_SIZE = {"crash": 905, "normal": 1905}
LEGACY_TEST_SIZE = 100

MANUSCRIPT_CLAIM = {
    "crash": ("2020-02-17", "2020-03-30"),
}

# Operational regime definitions, taken from the manuscript's own group table
# so that every model and every window is compared on the same dates.
REGIME_DEFINITION = {
    "crash": {
        "label": "COVID-19 crash",
        "start": "2020-02-17",
        "end": "2020-03-30",
        "definition": "Manuscript crash group G01-G15: first target 2020-02-17, "
                      "last target 2020-03-30, T = 31 forecast observations.",
        "source": "supplementary_backtest_version3.tex, group table lines 30-45",
    },
    "normal": {
        "label": "normal market (post-COVID)",
        "start": "2023-11-08",
        "end": "2024-04-03",
        "definition": "Manuscript normal group G06-G15 is the end of the sample "
                      "and is partial (T = 37 at W5, T = 12 at W30). The re-run "
                      "uses the full block that every window and stock can "
                      "support, 101 trading days, and reports the manuscript's "
                      "partial blocks next to it.",
        "source": "supplementary_backtest_version3.tex, group table lines 30-45",
    },
}

# the manuscript's own partial blocks, kept for comparison
MANUSCRIPT_PARTIAL = {
    ("normal", 5): ("2024-02-12", "2024-04-03", 37),
    ("normal", 30): ("2024-03-19", "2024-04-03", 12),
}

# --------------------------------------------------------------------------- #
# run defaults
# --------------------------------------------------------------------------- #
DEFAULT_SEED = 42
DEFAULT_VAL_DAYS = 50          # trading days of validation used for early stopping
DEFAULT_COST_BPS = [0, 10, 20, 40]
DEFAULT_CGAN_EPOCHS = 200
DEFAULT_CGAN_BATCH = 64
DEFAULT_LIGHTGBM_ROUNDS = 2000
DEFAULT_XGB_ROUNDS = 2000
