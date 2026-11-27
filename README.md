# run_sn_economics — leakage-free re-run of the SET50 forecasting experiments

Re-run of the SET50 cGAN and boosting experiments **without target leakage**, built to answer
reviewer comments **R3.M2** (data leakage in the public implementation) and **R3.M3**
(market-regime definitions and evaluation periods) of the manuscript
*“Conditional Generative Modelling of Thai Stock-Return Forecasts for Economic Applications”*.

Everything needed to repeat the run is in this repository: the program, the inputs,
the outputs and the published workbooks of the archived runs.

---

## 1. What was wrong with the archived run

The archived training script (`catBoost_conGAN.py`, and `new_cat_conGAN_modified.py`
for the per-stock SET50 runs) has three defects:

| lines | defect |
|---|---|
| 153–160 | `MinMaxScaler.fit_transform` is applied to **every sliding window before the split**, so the price range of the test period enters the training data |
| 165–170 | the split is index based (`train = all_indices[:905 + kk]`, `test = next 100`); there is **no validation block** |
| 234–240 | early stopping uses `eval_set=(X_test, y_test_orig)`, so the **test set drives the model selection** |

The regime blocks of the archived run are index offsets rather than dates, so the *normal*
block at W = 5 and W = 10 is the same file as the *crash* block, and the *normal* block is
truncated to the last rows of the sample.

## 2. What this re-run does

* strictly chronological splits, stated as **dates**: training → validation (50 trading days) → test;
* the scaler is fitted on the **training windows only**; the validation and test blocks are
  transformed with those constants;
* early stopping and the generator selection use the **validation block**; the test block is
  read once, by the frozen model;
* the same test dates for every look-back window inside a regime, so the cross-window
  comparisons rest on matched observations;
* every model is stored for every stock: CatBoost, LightGBM, XGBoost, the conditional GAN
  and a random-walk benchmark (the archived run only stored CatBoost, the cGAN and the hybrids).

## 3. The regime definitions

| regime | target dates | observations | source |
|---|---|---|---|
| crash | 2020-02-17 → 2020-03-30 | 31 trading days | group table G01–G15 of the supplementary material, which lists T = 31 |
| normal | 2023-11-08 → 2024-04-03 | 101 trading days | the largest block every window and stock can support |

Section 3.10 of the manuscript describes the crash evaluation as a 100-day evaluation running
from 17 February to 30 March 2020. That interval contains **31** trading days, and the group
table lists T = 31, so the two statements cannot both hold; the revision states the dates and
the number of observations and drops the 100-day description.

## 4. Repository layout

```
leakfree/               the prepared program
  config.py             paths, the 39 tickers, the regime definitions, the defaults
  data.py               reading the price workbooks (the price column is called
                        closed, close or cp depending on the file)
  splits.py             the chronological split and the training-only scaler
  models.py             CatBoost, LightGBM, XGBoost and the conditional GAN
  evaluation.py         the corrected backtest and Diebold-Mariano engine
  dm.py                 the DM test in the three conventions used in the literature
  portfolio.py          the equal-weight portfolio with a stationary bootstrap interval
  db.py                 PostgreSQL storage: results, chronology and a live progress table
  report.py             the auditable replication package
rerun_leakfree.py       the runner
verify_dm_convention.py reproduces the published DM summary matrix
make_response.py       turns a finished run into results_summary.json and the letter
publish_release.py     uploads the replication package to the server
backfill_results.py    re-inserts the result tables into the database after an interruption
input/                  the 39 price workbooks (the inputs of the run)
output/                 the results: forecasts, chronology, metrics, DM tests, provenance
results/                the summary and the response letter
archive/                the published workbooks of the archived runs
manifest.sha256         the SHA-256 of every file in this repository
```

## 5. How to run it again

```bash
python -m pip install -r requirements.txt

# the prices of this repository
export SET50_DATA_ROOT=input          # Windows: set SET50_DATA_ROOT=input

# the full grid: 39 stocks x 2 regimes x 3 windows
python rerun_leakfree.py --regimes crash normal --windows 5 10 30 \
       --models CatBoost LightGBM XGBoost cGAN --cgan-epochs 100 \
       --out outputs/rerun

# with the PostgreSQL results and the live progress table
python rerun_leakfree.py --regimes crash normal --windows 5 10 30 \
       --models CatBoost LightGBM XGBoost cGAN --cgan-epochs 100 \
       --db postgresql://user:password@host:5432/forecast_db --run-id my_run

# the summary and the response letter (the numbers come from the run)
python make_response.py --run outputs/rerun --suffix _final --pdf
```

The prices are read from the `input/` folder: the loader looks for `<ticker>_final2.xlsx`
in that folder and in a subfolder named after the ticker. The column that holds the closing
price is detected per file (`closed`, `close` or `cp`). The workbooks are stored flat, with
lower-case names, because `COM1`–`COM9` and `LPT1`–`LPT9` are reserved device names on
Windows (a path that contains them cannot be opened by git).

Expected runtime on 20 CPU cores: about two minutes per cell (the conditional GAN dominates,
100 epochs on the CPU), so roughly eight hours for the full grid of 234 cells. The boosting
models need one to two seconds per cell.

## 6. The published Diebold-Mariano matrix is reproduced exactly

`python verify_dm_convention.py --stocks ADVANC --publish` rebuilds the 3×3 matrix of
z-statistics from the archived `backtest_results_iter0.xlsx` files and compares it with the
published `dmtest_summary_matrix.xlsx`. The largest absolute difference over the nine cells
of the first stock is **4.4 × 10⁻¹⁶**, i.e. floating point noise, so the published Table 2 is
reproduced exactly by `leakfree/dm.py:dm_stat_paper`.

Three conventions are reported side by side, because the reviewer asks which one produced the
published numbers:

* `z_paper` — the published convention (`dmtest_summary_matrix.py`): population variance
  (ddof = 0), normal reference distribution, no small-sample correction;
* `z_matlab` — the convention of `dmtest.m` (Diebold-Mariano 1995, Ibisevic 2011): sample
  variance (ddof = 1), normal reference distribution, so `z_matlab = z_paper · √((T−1)/T)`;
* `dm_hln` — the corrected statistic: Newey-West Bartlett HAC, Harvey-Leybourne-Newbold
  correction, t(T−1) reference distribution, Holm adjustment inside each family.

Sign convention in all three: `d_t = loss_A − loss_B`, so a positive statistic means the first
model has the larger loss and is the worse one.

## 7. The headline result

The equal-weight portfolio over the stocks of each cell, 20 bps per unit of turnover, with a
stationary-bootstrap 95 % interval on the geometric return (see `results/results_summary.json`).

In the crash regime every model loses to the random walk and every interval covers zero; in
the calm regime the two larger windows are statistically indistinguishable from it. The mean of
the per-stock cumulative returns is **not** the return of a portfolio and is therefore not used
as the headline number; it is reported per stock in the results of the run.

## 8. Provenance

`output/report/provenance.json` records the SHA-256 of every input workbook, the versions of
the libraries, the hashes of the code files and the configuration of the run.
`manifest.sha256` holds the SHA-256 of every file in this repository, so the package can be
checked for tampering.

## 9. The package reproduces itself

A clean clone of this repository re-runs the cell ADVANC / crash / W5 and reproduces the
stored result **exactly**: the largest difference over the metrics (RMSE, MASE, cumulative
return, hit rate) is 0.0, and the chronology (train, validation and test dates and their
counts) is identical in every column.

The workflow `.github/workflows/reproduce.yml` does that on every push: it re-runs the cell
from the workbooks in `input/`, compares it with the row stored in `output/data/metrics_per_stock.csv`
and checks that every test block lies inside the definition of its regime.

## 10. The results on the server

The run writes to PostgreSQL (`forecast_db` on the server) and the tables, the progress table
and the replication package are served at

    https://srv2149219.hstgr.cloud:8452

with the pages *progress*, *tables* (the paper tables), *chronology* (per stock) and
*release* (the download of this package with its hashes).
