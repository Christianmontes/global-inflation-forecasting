"""
Case-study forecast results (Table 7).

Reads the per-country forecast-error workbooks of all models (main
1980-2019 sample, as in Table 2) and builds Table 7 for the five
case-study countries (United States, Nigeria, Brazil, Japan, United
Kingdom):

    columns (1)-(4)   RMSE, h = 1, 3, 6, 12
    columns (5)-(8)   MAD,  h = 1, 3, 6, 12

For each country, the RW row reports the level of the metric and the
remaining rows the ratio of the model's metric to the RW benchmark.
The ensemble is the equal-weighted combination of the EN, NN, RF, and
XGB forecasts, so its forecast error is the mean of those four error
series.

Known deviations from the published table (all other cells match):
  - Brazil, SRW, RMSE, h = 3: the data gives 1.182; the paper prints
    1.812, which appears to be a digit transposition.
  - Japan, RW level, RMSE, h = 12: the data gives 0.0012 (-> 0.001);
    the paper prints 0.003. The Japan h = 12 ratios all match, and
    they are computed from the same RW errors, so the printed level
    appears to be a typo.
  - Nigeria, NN, MAD, h = 1 and h = 12: the data gives 1.207 and
    0.832; the paper prints 1.201 and 0.835. The NN RMSE cells all
    match; like the AE cell in Table 11, the published numbers likely
    come from a slightly different run of the stochastic NN model than
    the stored error files.
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
BENCHMARK = "RW"
ENSEMBLE_OF = ["EN", "NN", "RF", "XGB"]
COUNTRIES = ["United States", "Nigeria", "Brazil", "Japan", "United Kingdom"]
MODELS = {
    "RW": "RW/e_RW_h{h}.xlsx",
    "SRW": "SRW/e_SRW_h{h}.xlsx",
    "AR": "AR dum/e_AR_dum_h{h}.xlsx",
    "CM": "CM model/e_CM_h{h}.xlsx",
    "EN": "Elastic net/e_EN_h{h}.xlsx",
    "NN": "NN/e_NN_h{h}.xlsx",
    "RF": "RF/e_RF_h{h}.xlsx",
    "XGB": "XGboost/e_XGB_h{h}.xlsx",
}

table = {}
for metric_name, metric in (("RMSE", rmse), ("MAD", mad)):
    for h in HORIZONS:
        errors = {m: pd.read_excel(ERRORS_DIR / f.format(h=h), index_col=0)[COUNTRIES]
                  for m, f in MODELS.items()}
        errors["Ensemble"] = sum(errors[m] for m in ENSEMBLE_OF) / len(ENSEMBLE_OF)
        values = pd.DataFrame({m: e.apply(metric) for m, e in errors.items()})
        col = values.div(values[BENCHMARK], axis=0)
        col[BENCHMARK] = values[BENCHMARK]  # benchmark column shows the level
        table[(metric_name, f"h = {h}")] = col.T.stack()

table = pd.DataFrame(table)
table = table.reorder_levels([1, 0]).loc[COUNTRIES]  # group rows by country
table.round(3).to_excel(RESULTS_DIR / "Table7_CaseStudies.xlsx")
print(table.round(3))
