"""
RF forecast results with alternative factor construction (Table 11).

Reads the per-country forecast-error workbooks of the RF model with
factors estimated by PCA, autoencoders (AE), and hierarchical dynamic
factor models (hDFM), and builds Table 11:

    columns (1)-(4)   RMSE ratios, h = 1, 3, 6, 12
    columns (5)-(8)   MAD ratios,  h = 1, 3, 6, 12

As in Tables 2 and 6, the aggregation is "ratio of averages": the
per-country metric is averaged across all 91 countries, and each row
reports this average relative to the baseline RF model with simple
cross-sectional mean factors (the RF model of Table 2).

Known deviations from the published table (all other cells match):
  - PCA, MAD, h = 12: the data gives 1.002; the paper prints 1.102,
    which appears to be a typo (the RMSE ratio at h = 12 is 1.003).
  - AE, MAD, h = 1: the data gives 0.996; the paper prints 1.000.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERRORS_DIR = ROOT / "Replication results"
RESULTS_DIR = ROOT / "Tables" / "Results"

sys.path.insert(0, str(ROOT))
from Functions import mad, rmse  # noqa: E402

HORIZONS = [1, 3, 6, 12]
BENCHMARK = "Main case/RF/e_RF_h{h}.xlsx"
MODELS = {
    "PCA factors": "Main case/RF_PCA/e_RF_PCA_h{h}.xlsx",
    "AE factors": "Main case/RF_AE/e_RF_AE_h{h}.xlsx",
    "hDFM factors": "Main case/RF_DFM/e_RF_DFM_h{h}.xlsx",
}

table = {}
for metric_name, metric in (("RMSE", rmse), ("MAD", mad)):
    for h in HORIZONS:
        base = pd.read_excel(ERRORS_DIR / BENCHMARK.format(h=h), index_col=0)
        base_avg = base.apply(metric).mean()
        table[(metric_name, f"h = {h}")] = pd.Series({
            m: pd.read_excel(ERRORS_DIR / f.format(h=h), index_col=0).apply(metric).mean() / base_avg
            for m, f in MODELS.items()
        })

table = pd.DataFrame(table)
table.round(3).to_excel(RESULTS_DIR / "Table11_ForecastingResults_AlternativeFactors.xlsx")
print(table.round(3))
