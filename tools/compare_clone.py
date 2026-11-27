"""Compare the result of a fresh clone with the results stored in the repository."""
import sys
from pathlib import Path

import pandas as pd

clone = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\tmp\clone_test_sn")
new_metrics = pd.read_csv(clone / "outputs" / "smoke" / "data" / "metrics_per_stock.csv")
old_metrics = pd.read_csv(r"D:\run_sn_economics\output\data\metrics_per_stock.csv")

new = new_metrics[(new_metrics.leakage == "clean") & (new_metrics.cost_bps == 20)]
old = old_metrics[(old_metrics.leakage == "clean") & (old_metrics.cost_bps == 20)
                   & (old_metrics.ticker == "ADVANC") & (old_metrics.regime == "crash")
                   & (old_metrics.window == 5)]
key = ["ticker", "regime", "window", "model", "cost_bps"]
merged = new.merge(old, on=key, suffixes=("_new", "_old"))
print("  ช่องที่เทียบ:", len(merged))

worst = 0.0
for _, row in merged.iterrows():
    for column in ("rmse", "mase", "cum_return_geom", "hit_rate"):
        worst = max(worst, abs(float(row[column + "_new"]) - float(row[column + "_old"])))
    print("    {:11} RMSE {:.6f} vs {:.6f} | MASE {:.6f} vs {:.6f} | ret {:.6f} vs {:.6f}".format(
        row["model"], row["rmse_new"], row["rmse_old"], row["mase_new"], row["mase_old"],
        row["cum_return_geom_new"], row["cum_return_geom_old"]))
print("  ความต่างสูงสุดเทียบของเดิม: {:.3e}".format(worst))
print("  ผลตรงกัน:", worst < 1e-9)

chron_new = pd.read_csv(clone / "outputs" / "smoke" / "data" / "chronology.csv")
chron_old = pd.read_csv(r"D:\run_sn_economics\output\data\chronology.csv")
columns = ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
           "n_train", "n_val", "n_test"]
merged = chron_new.merge(chron_old, on=["ticker", "regime", "window"],
                          suffixes=("_new", "_old"))
same = all((merged[column + "_new"].astype(str) == merged[column + "_old"].astype(str)).all()
           for column in columns)
print("  chronology ตรงกันทุกคอลัมน์:", same)
