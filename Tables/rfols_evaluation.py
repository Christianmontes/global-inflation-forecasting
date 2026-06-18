"""
RF/OLS forecast results (Table 6).

Reads the per-country forecast-error workbooks of the RF/OLS, EN, and
RF models (main 1980-2019 sample) and builds Table 6:

    columns (1)-(4)   RMSE, h = 1, 3, 6, 12
    columns (5)-(8)   MAD,  h = 1, 3, 6, 12

As in Table 2, the aggregation is "ratio of averages": the per-country
metric is averaged across all 91 countries, and the EN and RF rows
report this average relative to the RF/OLS model, whose own row shows
the average level.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERRORS_DIR = ROOT / "Replication results" / "Main case"
RESULTS_DIR = ROOT / "Tables" / "Results"

sys.path.insert(0, str(ROOT))
from Functions import mad, rmse  # noqa: E402

HORIZONS = [1, 3, 6, 12]
BENCHMARK = "RF/OLS"
MODELS = {
    "RF/OLS": "RFOLS/e_RFOLS_h{h}.xlsx",
    "EN": "Elastic net/e_EN_h{h}.xlsx",
    "RF": "RF/e_RF_h{h}.xlsx",
}

table = {}
for metric_name, metric in (("RMSE", rmse), ("MAD", mad)):
    for h in HORIZONS:
        errors = {m: pd.read_excel(ERRORS_DIR / f.format(h=h), index_col=0)
                  for m, f in MODELS.items()}
        avg = pd.Series({m: e.apply(metric).mean() for m, e in errors.items()})
        col = avg / avg[BENCHMARK]
        col[BENCHMARK] = avg[BENCHMARK]  # benchmark row shows the level
        table[(metric_name, f"h = {h}")] = col

table = pd.DataFrame(table)
table.round(3).to_excel(RESULTS_DIR / "Table6_ForecastingResults_RFOLS.xlsx")
print(table.round(3))
