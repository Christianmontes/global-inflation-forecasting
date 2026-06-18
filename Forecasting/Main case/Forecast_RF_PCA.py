"""Random Forest forecast with PCA factors.

Same as Forecast_RF.py but the global and regional factors are the first principal
component of the (standardised) diffed inflation panels instead of the
cross-sectional mean. For each rolling window the panel columns are standardised
and PCA is fit on the whole window; the first component is used as the factor.

Rolling-window, expanding out-of-sample forecasts for every country and horizon
h in {1, 3, 6, 12}, 20-year window. Results are written to
'Replication results/Main case/RF_PCA' as Fcast_RF_PCA_h{h}.xlsx / e_RF_PCA_h{h}.xlsx.
This variant saves no SHAP values or training frames (only RF-mean does).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _diff, _lags, _pca, _rfTune, _rfPred, load_panels, make_month_dummies

DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
OUT_DIR = ROOT / "Replication results" / "Main case" / "RF_PCA"

max_lags = 4
win1 = 12 * 20            # rolling window length (~20 years -> forecasts from 2000)
horizon_vector = [1, 3, 6, 12]
model_config_rf = {
    "n_estimators": 500,
    "grid": {"max_features": [None, 2 / 3, 1 / 3, "sqrt", "log2"]},
}


data, continent_row = load_panels(DATA_FILE)
dummies = make_month_dummies(data['Global'].index, drop=None)
ncol = data['Global'].shape[1]

for h in horizon_vector:
    win2 = win1 + h

    shifted_data = {region: df.shift(periods=h).dropna() for region, df in data.items()}
    shifted_dummies = dummies.shift(periods=h).dropna()
    n_rows = len(shifted_data['Global'])

    fc = np.full((n_rows, ncol), np.nan)
    er = np.full((n_rows, ncol), np.nan)

    for i in range(ncol):
        print(f"h={h} country {i}")

        region_data = shifted_data.get(str(continent_row.iloc[i]).replace(" ", ""))
        data_temp = data['Global'].iloc[1:, i]
        ret = data_temp.rolling(h).mean().dropna().to_frame("Returns")

        best_params = None  # set at the first window and each December
        for t in range(win2, len(ret) + 1):
            train_global = _diff(shifted_data['Global'].iloc[t - win2:t])
            train_region = _diff(region_data.iloc[t - win2:t])

            # PCA factors (first component of the standardised window)
            pca_global = _pca(train_global, "MeanGlobal")
            pca_region = _pca(train_region, "MeanRegion")

            train_global_excl = train_global.drop(columns=data_temp.name, errors='ignore')
            train_ret = ret.iloc[t - win2:t - h]
            test_ret = ret.iloc[t - 1].to_frame().T

            train_slices = {
                'Ret': train_ret,
                'Dummies': shifted_dummies.iloc[t - win2:t - h],
                'Inflation': train_global_excl.iloc[:-h],
                'Lags': _lags(data_temp.iloc[t - win2:t - h], max_lags).set_index(train_ret.index[max_lags:]),
                'GlobalFactor': pca_global.iloc[:-h],
                'RegionFactor': pca_region.iloc[:-h],
            }
            test_slices = {
                'Ret': test_ret,
                'Dummies': shifted_dummies.iloc[t - 1, :].to_frame().T,
                'Inflation': train_global_excl.iloc[-1].to_frame().T,
                'Lags': _lags(data_temp.iloc[t - win2:t], max_lags).iloc[-1:].set_index(test_ret.index),
                'GlobalFactor': pca_global.iloc[-1].to_frame().T,
                'RegionFactor': pca_region.iloc[-1].to_frame().T,
            }

            train = pd.concat(train_slices.values(), axis=1).dropna()
            test = pd.concat(test_slices.values(), axis=1).dropna()

            if (t == win2) or (pd.to_datetime(test_ret.index[0]).month == 12) or (best_params is None):
                best_params = _rfTune(train, 'Returns', h, model_config_rf)

            yhat_val, _ = _rfPred(train, test, 'Returns', best_params, model_config_rf)
            fc[t - 1, i] = yhat_val
            er[t - 1, i] = test['Returns'].iloc[0] - yhat_val

    idx, cols = shifted_data['Global'].index, shifted_data['Global'].columns
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fc, index=idx, columns=cols).to_excel(OUT_DIR / f"Fcast_RF_PCA_h{h}.xlsx")
    pd.DataFrame(er, index=idx, columns=cols).to_excel(OUT_DIR / f"e_RF_PCA_h{h}.xlsx")
    print(f"h={h}: wrote RF PCA to {OUT_DIR}")
