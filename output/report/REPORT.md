# Leakage-free re-run of the SET50 forecasting experiments

Run created 2026-11-26T09:40:57+00:00.

## 1. What changed with respect to the archived implementation

| Item | Archived run | This re-run |
|---|---|---|
| Normalisation | `MinMaxScaler.fit_transform` on **all** sliding windows before the split (`new_cat_conGAN_modified.py` lines 153-160) | scaler fitted on the **training windows only**; validation and test are transformed with those constants |
| Split | index offsets `train = 905 + kk`, `test = next 100` (lines 165-170) | chronological split stated as **dates**: training, then a validation block, then the untouched test block |
| Early stopping | `eval_set=(X_test, y_test_orig)` (lines 234-240) | `eval_set=(X_val, y_val)`; the test block is read once, at the end |
| Generator selection | last epoch | epoch with the best **validation** loss |
| Saved predictions | per stock only for CatBoost, cGAN and the hybrids; LightGBM and XGBoost were never stored | every model stored per stock, per regime, per window |

## 2. The test blocks actually used

| regime   |   window |   stocks |   n_test_median |   n_test_min |   n_test_max | test_start   | test_end   |
|:---------|---------:|---------:|----------------:|-------------:|-------------:|:-------------|:-----------|
| crash    |        5 |       39 |              31 |           31 |           31 | 2020-02-17   | 2020-03-30 |
| crash    |       10 |       39 |              31 |           31 |           31 | 2020-02-17   | 2020-03-30 |
| crash    |       30 |       39 |              31 |           31 |           31 | 2020-02-17   | 2020-03-30 |
| normal   |        5 |       39 |             101 |          101 |          101 | 2023-11-08   | 2024-04-03 |
| normal   |       10 |       39 |             101 |          101 |          101 | 2023-11-08   | 2024-04-03 |
| normal   |       30 |       39 |             101 |          101 |          101 | 2023-11-08   | 2024-04-03 |

The archived 'normal' regime used `train_size = 1905` on a sample that ends on 3 April 2024, so its test block is truncated to the last few rows of the file. The re-run reports the same truncation instead of hiding it.

## 3. Regime labels

| regime   |   window |   stocks |   inside |
|:---------|---------:|---------:|---------:|
| crash    |        5 |       39 |       39 |
| crash    |       10 |       39 |       39 |
| crash    |       30 |       39 |       39 |
| normal   |        5 |       39 |       39 |
| normal   |       10 |       39 |       39 |
| normal   |       30 |       39 |       39 |

- **crash** (COVID-19 crash): 2020-02-17 to 2020-03-30. Manuscript crash group G01-G15: first target 2020-02-17, last target 2020-03-30, T = 31 forecast observations. Source: supplementary_backtest_version3.tex, group table lines 30-45.
- **normal** (normal market (post-COVID)): 2023-11-08 to 2024-04-03. Manuscript normal group G06-G15 is the end of the sample and is partial (T = 37 at W5, T = 12 at W30). The re-run uses the full block that every window and stock can support, 101 trading days, and reports the manuscript's partial blocks next to it. Source: supplementary_backtest_version3.tex, group table lines 30-45.

The manuscript describes the crash evaluation in Section 3.10 as 100 days from 17 February to 30 March 2020. That calendar interval contains 31 trading days, and the manuscript's own group table lists T = 31 for every crash group, so the two statements cannot both hold. The re-run uses the group table, i.e. the dates, and reports the number of observations that the data actually supports.

## 4. Matched dates and exclusions

| regime   |   window |   stocks |   matched_min |   matched_max |
|:---------|---------:|---------:|--------------:|--------------:|
| crash    |        5 |       39 |            31 |            31 |
| crash    |       10 |       39 |            31 |            31 |
| crash    |       30 |       39 |            31 |            31 |
| normal   |        5 |       39 |           101 |           101 |
| normal   |       10 |       39 |           101 |           101 |
| normal   |       30 |       39 |           101 |           101 |

## 5. Effect of removing the leak

| metric          |   mean_clean |   mean_legacy |   mean_difference |   median_difference |   n_pairs |   n_clean_better |
|:----------------|-------------:|--------------:|------------------:|--------------------:|----------:|-----------------:|
| cum_return_geom |   -0.0781771 |    -0.0160215 |        -0.0621556 |          -0.0129476 |      3744 |              871 |
| sharpe_annual   |   -0.766712  |     0.270457  |        -1.03717   |          -0.136285  |      3744 |              886 |
| mase            |    2.76717   |     2.42337   |         0.343807  |           0.0378666 |      3744 |             2316 |
| rmse            |    4.71711   |     3.32788   |         1.38923   |           0.325567  |      3744 |             3308 |
| hit_rate        |    0.331692  |     0.347254  |        -0.0155626 |           0         |      3744 |             1168 |

The clean rule is the one that should be reported. The archived rule is kept in the release only so the difference can be inspected rather than asserted.

## 6. Accuracy and trading results (clean run)

| regime   |   window | model      |   n |   mean_return |   mean_mase |   mean_rmse |   mean_hit |
|:---------|---------:|:-----------|----:|--------------:|------------:|------------:|-----------:|
| crash    |        5 | CatBoost   | 156 |       -0.1703 |      3.9217 |      8.3752 |     0.4718 |
| crash    |        5 | LightGBM   | 156 |       -0.1207 |      3.8713 |      8.3628 |     0.4786 |
| crash    |        5 | RandomWalk | 156 |        0      |      0.9803 |      2.781  |     0      |
| crash    |        5 | XGBoost    | 156 |       -0.1107 |      4.0403 |      8.3947 |     0.4803 |
| crash    |        5 | cGAN       | 156 |       -0.1221 |      5.1565 |     11.6763 |     0.4513 |
| crash    |       10 | CatBoost   | 156 |       -0.1922 |      4.233  |      9.2913 |     0.4581 |
| crash    |       10 | LightGBM   | 156 |       -0.163  |      3.9947 |      8.661  |     0.4701 |
| crash    |       10 | RandomWalk | 156 |        0      |      0.9803 |      2.781  |     0      |
| crash    |       10 | XGBoost    | 156 |       -0.1283 |      4.358  |      9.2318 |     0.4709 |
| crash    |       10 | cGAN       | 156 |       -0.1586 |      5.2796 |     12.1345 |     0.4479 |
| crash    |       30 | CatBoost   | 156 |       -0.2204 |      4.9056 |     11.041  |     0.4427 |
| crash    |       30 | LightGBM   | 156 |       -0.1701 |      4.27   |      9.5215 |     0.4658 |
| crash    |       30 | RandomWalk | 156 |        0      |      0.9803 |      2.781  |     0      |
| crash    |       30 | XGBoost    | 156 |       -0.0972 |      4.6908 |     10.3399 |     0.4726 |
| crash    |       30 | cGAN       | 156 |       -0.1547 |      5.5454 |     12.1983 |     0.4547 |
| normal   |        5 | CatBoost   | 156 |       -0.0372 |      2.2381 |      1.8654 |     0.4246 |
| normal   |        5 | LightGBM   | 156 |       -0.1116 |      2.256  |      1.9066 |     0.4018 |
| normal   |        5 | RandomWalk | 156 |        0      |      1.0013 |      0.8532 |     0      |
| normal   |        5 | XGBoost    | 156 |       -0.1108 |      2.3757 |      2.0806 |     0.4069 |
| normal   |        5 | cGAN       | 156 |       -0.1502 |      9.7611 |      7.7198 |     0.3979 |
| normal   |       10 | CatBoost   | 156 |       -0.0201 |      2.4297 |      2.0661 |     0.4254 |
| normal   |       10 | LightGBM   | 156 |       -0.0755 |      2.2684 |      1.9306 |     0.4144 |
| normal   |       10 | RandomWalk | 156 |        0      |      1.0013 |      0.8532 |     0      |
| normal   |       10 | XGBoost    | 156 |       -0.0502 |      2.4946 |      2.2216 |     0.4213 |
| normal   |       10 | cGAN       | 156 |       -0.1248 |      9.9378 |      7.5633 |     0.3918 |
| normal   |       30 | CatBoost   | 156 |       -0.004  |      2.9302 |      2.5083 |     0.429  |
| normal   |       30 | LightGBM   | 156 |       -0.0495 |      2.4202 |      2.0722 |     0.411  |
| normal   |       30 | RandomWalk | 156 |        0      |      1.0013 |      0.8532 |     0      |
| normal   |       30 | XGBoost    | 156 |       -0.0444 |      2.7691 |      2.4377 |     0.4151 |
| normal   |       30 | cGAN       | 156 |       -0.1223 |     10.8708 |      7.813  |     0.4038 |

## 7. Diebold-Mariano tests

`d_t = loss_A - loss_B`, so a positive statistic means the first model is the worse one. Newey-West HAC variance at lag h-1, Harvey-Leybourne-Newbold correction, t(T-1) reference distribution, Holm adjustment inside each family.

| regime   |   window | model_A    | model_B    |   tests |    mean_d |   significant_holm |
|:---------|---------:|:-----------|:-----------|--------:|----------:|-------------------:|
| crash    |        5 | CatBoost   | LightGBM   |      39 |  -28.2843 |                 16 |
| crash    |        5 | CatBoost   | RandomWalk |      39 |  203.254  |                 17 |
| crash    |        5 | CatBoost   | XGBoost    |      39 |  -15.578  |                 16 |
| crash    |        5 | LightGBM   | CatBoost   |      39 |   28.2843 |                 16 |
| crash    |        5 | LightGBM   | RandomWalk |      39 |  231.538  |                 16 |
| crash    |        5 | LightGBM   | XGBoost    |      39 |   12.7063 |                 16 |
| crash    |        5 | RandomWalk | CatBoost   |      39 | -203.254  |                 17 |
| crash    |        5 | RandomWalk | LightGBM   |      39 | -231.538  |                 16 |
| crash    |        5 | RandomWalk | XGBoost    |      39 | -218.832  |                 17 |
| crash    |        5 | XGBoost    | CatBoost   |      39 |   15.578  |                 16 |
| crash    |        5 | XGBoost    | LightGBM   |      39 |  -12.7063 |                 16 |
| crash    |        5 | XGBoost    | RandomWalk |      39 |  218.832  |                 17 |
| crash    |        5 | cGAN       | CatBoost   |      39 |   84.2909 |                 26 |
| crash    |        5 | cGAN       | LightGBM   |      39 |   56.0065 |                 27 |
| crash    |        5 | cGAN       | RandomWalk |      39 |  287.545  |                 36 |
| crash    |        5 | cGAN       | XGBoost    |      39 |   68.7129 |                 26 |
| crash    |       10 | CatBoost   | LightGBM   |      39 |   10.9922 |                 17 |
| crash    |       10 | CatBoost   | RandomWalk |      39 |  246.474  |                 22 |
| crash    |       10 | CatBoost   | XGBoost    |      39 |  -21.7095 |                 21 |
| crash    |       10 | LightGBM   | CatBoost   |      39 |  -10.9922 |                 17 |
| crash    |       10 | LightGBM   | RandomWalk |      39 |  235.482  |                 17 |
| crash    |       10 | LightGBM   | XGBoost    |      39 |  -32.7017 |                 16 |
| crash    |       10 | RandomWalk | CatBoost   |      39 | -246.474  |                 22 |
| crash    |       10 | RandomWalk | LightGBM   |      39 | -235.482  |                 17 |
| crash    |       10 | RandomWalk | XGBoost    |      39 | -268.183  |                 18 |
| crash    |       10 | XGBoost    | CatBoost   |      39 |   21.7095 |                 21 |
| crash    |       10 | XGBoost    | LightGBM   |      39 |   32.7017 |                 16 |
| crash    |       10 | XGBoost    | RandomWalk |      39 |  268.183  |                 18 |
| crash    |       10 | cGAN       | CatBoost   |      39 |   78.4768 |                 30 |
| crash    |       10 | cGAN       | LightGBM   |      39 |   89.469  |                 31 |
| crash    |       10 | cGAN       | RandomWalk |      39 |  324.951  |                 36 |
| crash    |       10 | cGAN       | XGBoost    |      39 |   56.7673 |                 31 |
| crash    |       30 | CatBoost   | LightGBM   |      39 |   57.5321 |                 29 |
| crash    |       30 | CatBoost   | RandomWalk |      39 |  342.047  |                 30 |
| crash    |       30 | CatBoost   | XGBoost    |      39 |   10.4943 |                 27 |
| crash    |       30 | LightGBM   | CatBoost   |      39 |  -57.5321 |                 29 |
| crash    |       30 | LightGBM   | RandomWalk |      39 |  284.515  |                 22 |
| crash    |       30 | LightGBM   | XGBoost    |      39 |  -47.0378 |                 22 |
| crash    |       30 | RandomWalk | CatBoost   |      39 | -342.047  |                 30 |
| crash    |       30 | RandomWalk | LightGBM   |      39 | -284.515  |                 22 |
| crash    |       30 | RandomWalk | XGBoost    |      39 | -331.553  |                 21 |
| crash    |       30 | XGBoost    | CatBoost   |      39 |  -10.4943 |                 27 |
| crash    |       30 | XGBoost    | LightGBM   |      39 |   47.0378 |                 22 |
| crash    |       30 | XGBoost    | RandomWalk |      39 |  331.553  |                 21 |
| crash    |       30 | cGAN       | CatBoost   |      39 |  -43.5547 |                 20 |
| crash    |       30 | cGAN       | LightGBM   |      39 |   13.9775 |                 27 |
| crash    |       30 | cGAN       | RandomWalk |      39 |  298.492  |                 39 |
| crash    |       30 | cGAN       | XGBoost    |      39 |  -33.0604 |                 33 |
| normal   |        5 | CatBoost   | LightGBM   |      39 |   -1.9295 |                  9 |
| normal   |        5 | CatBoost   | RandomWalk |      39 |   17.6946 |                 14 |
| normal   |        5 | CatBoost   | XGBoost    |      39 |  -11.8005 |                 10 |
| normal   |        5 | LightGBM   | CatBoost   |      39 |    1.9295 |                  9 |
| normal   |        5 | LightGBM   | RandomWalk |      39 |   19.6241 |                 19 |
| normal   |        5 | LightGBM   | XGBoost    |      39 |   -9.871  |                  9 |
| normal   |        5 | RandomWalk | CatBoost   |      39 |  -17.6946 |                 14 |
| normal   |        5 | RandomWalk | LightGBM   |      39 |  -19.6241 |                 19 |
| normal   |        5 | RandomWalk | XGBoost    |      39 |  -29.4952 |                 14 |
| normal   |        5 | XGBoost    | CatBoost   |      39 |   11.8005 |                 10 |
| normal   |        5 | XGBoost    | LightGBM   |      39 |    9.871  |                  9 |
| normal   |        5 | XGBoost    | RandomWalk |      39 |   29.4952 |                 14 |
| normal   |        5 | cGAN       | CatBoost   |      39 |   99.9099 |                 37 |
| normal   |        5 | cGAN       | LightGBM   |      39 |   97.9804 |                 37 |
| normal   |        5 | cGAN       | RandomWalk |      39 |  117.605  |                 37 |
| normal   |        5 | cGAN       | XGBoost    |      39 |   88.1094 |                 37 |
| normal   |       10 | CatBoost   | LightGBM   |      39 |    2.6781 |                 13 |
| normal   |       10 | CatBoost   | RandomWalk |      39 |   23.3946 |                 17 |
| normal   |       10 | CatBoost   | XGBoost    |      39 |  -13.823  |                 11 |
| normal   |       10 | LightGBM   | CatBoost   |      39 |   -2.6781 |                 13 |
| normal   |       10 | LightGBM   | RandomWalk |      39 |   20.7166 |                 17 |
| normal   |       10 | LightGBM   | XGBoost    |      39 |  -16.501  |                  7 |
| normal   |       10 | RandomWalk | CatBoost   |      39 |  -23.3946 |                 17 |
| normal   |       10 | RandomWalk | LightGBM   |      39 |  -20.7166 |                 17 |
| normal   |       10 | RandomWalk | XGBoost    |      39 |  -37.2176 |                 15 |
| normal   |       10 | XGBoost    | CatBoost   |      39 |   13.823  |                 11 |
| normal   |       10 | XGBoost    | LightGBM   |      39 |   16.501  |                  7 |
| normal   |       10 | XGBoost    | RandomWalk |      39 |   37.2176 |                 15 |
| normal   |       10 | cGAN       | CatBoost   |      39 |   81.8409 |                 36 |
| normal   |       10 | cGAN       | LightGBM   |      39 |   84.519  |                 36 |
| normal   |       10 | cGAN       | RandomWalk |      39 |  105.236  |                 38 |
| normal   |       10 | cGAN       | XGBoost    |      39 |   68.0179 |                 36 |
| normal   |       30 | CatBoost   | LightGBM   |      39 |   14.1991 |                 18 |
| normal   |       30 | CatBoost   | RandomWalk |      39 |   40.6557 |                 24 |
| normal   |       30 | CatBoost   | XGBoost    |      39 |   -7.0184 |                 15 |
| normal   |       30 | LightGBM   | CatBoost   |      39 |  -14.1991 |                 18 |
| normal   |       30 | LightGBM   | RandomWalk |      39 |   26.4566 |                 19 |
| normal   |       30 | LightGBM   | XGBoost    |      39 |  -21.2175 |                 10 |
| normal   |       30 | RandomWalk | CatBoost   |      39 |  -40.6557 |                 24 |
| normal   |       30 | RandomWalk | LightGBM   |      39 |  -26.4566 |                 19 |
| normal   |       30 | RandomWalk | XGBoost    |      39 |  -47.6741 |                 18 |
| normal   |       30 | XGBoost    | CatBoost   |      39 |    7.0184 |                 15 |
| normal   |       30 | XGBoost    | LightGBM   |      39 |   21.2175 |                 10 |
| normal   |       30 | XGBoost    | RandomWalk |      39 |   47.6741 |                 18 |
| normal   |       30 | cGAN       | CatBoost   |      39 |   72.0882 |                 36 |
| normal   |       30 | cGAN       | LightGBM   |      39 |   86.2873 |                 36 |
| normal   |       30 | cGAN       | RandomWalk |      39 |  112.744  |                 39 |
| normal   |       30 | cGAN       | XGBoost    |      39 |   65.0698 |                 37 |

## 8. How to reproduce

```bash
python rerun_leakfree.py --out outputs/<run> --models CatBoost LightGBM XGBoost cGAN \
       --regimes crash normal --windows 5 10 30 --val-days 50
```

`report/provenance.json` holds the hash of every input file, of every module of this program and of the archived evaluation engine, together with the library versions and the random seed. `manifest.sha256` holds the hash of every file in the release.
