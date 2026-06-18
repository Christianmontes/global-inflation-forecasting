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

This script reproduces the published (final-submission) Table 11 exactly. Two
typos in the originally submitted version (PCA MAD h = 12; AE MAD h = 1) were
corrected in the final published table and now agree with the values produced
here.
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
