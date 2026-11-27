# Leakage-free re-run of the SET50 forecasting experiments

Re-run of the SET50 cGAN and boosting experiments without target leakage, built to
answer reviewer comments R3.M2 and R3.M3 of the manuscript on the SET50 cGAN study.

## What was wrong with the archived run

The archived training script (`D:\CatBoost\<TICKER>\new_cat_conGAN_modified.py`, and
`D:\CatBoost\catBoost_conGAN.py` for the DAX version) has three defects:

| line | defect |
|---|---|
| 153-160 | `MinMaxScaler.fit_transform` is applied to **every** sliding window before the split, so the price range of the test period enters the training data |
| 165-170 | the split is `train = all_indices[:905 + kk]`, `test = next 100`; there is **no validation block** |
| 234-240 | early stopping uses `eval_set=(X_test, y_test_orig)`, so the **test set drives model selection** |

## What this program does

```
leakfree/config.py    paths, the 39 tickers, the regime definitions, the defaults
leakfree/data.py      reading the archived workbooks (the price column is called
                      closed, close or cp depending on the file)
leakfree/splits.py    chronological split by dates, scaler fitted on training only
leakfree/models.py    CatBoost, LightGBM, XGBoost and the conditional GAN, all with
                      early stopping on the validation block
leakfree/dm.py        the Diebold-Mariano test in the three conventions in use here
leakfree/evaluation.py  a byte-identical copy of D:\run_revised_conGANs\revised_core.py
leakfree/db.py        PostgreSQL storage: results, chronology and a live progress table
leakfree/report.py    the auditable replication package
rerun_leakfree.py     the runner
finish_run.py         adds the models missing from a run's stored forecasts, writes the rest
leakfree/r3m4.py      the R3.M4 analyses: worked example, pooled DM tests, lag-robust counts
verify_dm_convention.py  reproduces the published DM summary matrix
webapp/               the website that shows the tables and the progress
```

## Reproduce

```bash
python rerun_leakfree.py --regimes crash normal --windows 5 10 30 \
       --models CatBoost LightGBM XGBoost cGAN --cgan-epochs 100 \
       --db postgresql://user:password@host:5432/forecast_db
```

The run writes, per cell, to `data/` (chronology, predictions, metrics, DM tests,
coverage, leakage effect) and to the database, and finishes with
`report/REPORT.md`, `report/provenance.json` and `manifest.sha256`.

## Finishing a run

`finish_run.py` adds the models that are missing from the stored forecasts of a run
(the cGAN of `run_20261126_060638`, whose first session stopped after the boosting
models) and then writes everything that depends on all models: the metrics of the
added models, the DM tests, Table 2, the chronology, coverage, provenance, report and
manifest. It resumes: cells that already hold the model are skipped.

```bash
python finish_run.py --run outputs/run_20261126_060638 --cgan-epochs 100   # train, then finish
python finish_run.py --run outputs/run_20261126_060638 --skip-train        # finish only
```

Training sessions can run side by side on disjoint `--tickers` with `--only-train`.
Every trained cell writes `data/training_log/<cell>.json` (best epoch, validation
RMSE, loss history), and `data/training_environment.json` records the libraries used.
`make_response.py --run outputs/run_20261126_060638 --pdf` then writes the response
letter `answer_sn_economics6.tex`, with the R3.M4 analyses of `leakfree/r3m4.py`
(`data/r3m4/`).

## The published Diebold-Mariano matrix is reproduced exactly

`verify_dm_convention.py --stocks ADVANC --publish` rebuilds the 3x3 matrix of
z-statistics from the archived `backtest_results_iter0.xlsx` files and compares it
with `C:\program_stock\dmtest_summary_matrix.xlsx`. The largest absolute difference
is 4.4e-16, so the published Table 2 is reproduced exactly by
`leakfree/dm.dm_stat_paper`.

Three conventions are reported side by side, because the reviewer asked which one
produced the published numbers:

* `z_paper` - the published convention (`dmtest_summary_matrix.py`): population
  variance (ddof = 0), normal reference distribution, no small-sample correction.
* `z_matlab` - the convention of `dmtest.m` (Diebold-Mariano 1995, Ibisevic 2011):
  sample variance (ddof = 1), normal reference distribution.
* `dm_hln` - the corrected statistic: Newey-West Bartlett HAC, Harvey-Leybourne-Newbold
  correction, t(T-1) reference distribution, Holm adjustment inside each family.

Sign convention in all three: `d_t = loss_A - loss_B`, so a positive statistic means
the first model is the worse one.

## Regime definitions

| regime | target dates | source |
|---|---|---|
| crash | 2020-02-17 to 2020-03-30 (31 trading days) | the manuscript's own group table, `supplementary_backtest_version3.tex` lines 30-45 |
| normal | 2023-11-08 to 2024-04-03 (101 trading days) | the largest block every window and stock can support; the manuscript's own normal groups are partial (T = 37 at W5, T = 12 at W30) |

The manuscript describes the crash evaluation in Section 3.10 as 100 days from
17 February to 30 March 2020. That interval holds 31 trading days, and the
manuscript's group table lists T = 31, so the two statements cannot both hold. The
re-run uses the dates and reports the number of observations the data supports.

## Website

The results and the live progress table are served by `webapp/` (FastAPI in Docker)
at `https://srv2149219.hstgr.cloud:8452`, backed by the PostgreSQL database
`forecast_db` on the same server.