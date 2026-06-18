"""Random Forest forecast with hierarchical dynamic-factor-model (DFM) factors.

Same as Forecast_RF.py but the global and regional factors come from a
hierarchical dynamic factor model (statsmodels) instead of the cross-sectional
mean: a global DFM is fit on the standardised diffed panel, the panel is
deflated by the train-OLS loadings on the global factor, and a regional DFM is
fit on the residual panel restricted to the region's columns. The global and
regional factors (train-only standardised) replace the mean factors. Everything
else (features, ordering, tuning schedule) is identical to Forecast_RF.py.

Rolling-window, expanding out-of-sample forecasts for every country and horizon
h in {1, 3, 6, 12}, 20-year window. Results are written to
'Replication results/Main case/RF_DFM' as Fcast_RF_DFM_h{h}.xlsx / e_RF_DFM_h{h}.xlsx.

Note: the DFM is fit by EM/MLE (statsmodels DynamicFactor), which is markedly
more version-sensitive than the mean/PCA factors, so exact replication of the
stored DFM results across statsmodels versions is not expected.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _diff, _lags, _rfTune, _rfPred, load_panels, make_month_dummies
from Functions import HierarchicalDFMExtractor

DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
OUT_DIR = ROOT / "Replication results" / "Main case" / "RF_DFM"

max_lags = 4
win1 = 12 * 20            # rolling window length (~20 years -> forecasts from 2000)
horizon_vector = [1, 3, 6, 12]
model_config_rf = {
    "n_estimators": 500,
    "grid": {"max_features": [None, 2 / 3, 1 / 3, "sqrt", "log2"]},
}

# Hierarchical DFM extractor (global -> deflate -> regional), instantiated once.
hdfm = HierarchicalDFMExtractor(
    factor_order=1,
    error_cov_type="diagonal",
    use_smoothed=True,
    maxiter=500,
    em_iter=25,
    zscore_residuals=True,
    prune_near_constant=True,
)

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

        region_key = str(continent_row.iloc[i]).replace(" ", "")
        region_data = shifted_data.get(region_key)
        data_temp = data['Global'].iloc[1:, i]
        ret = data_temp.rolling(h).mean().dropna().to_frame("Returns")

        best_params = None  # set at the first window and each December
        for t in range(win2, len(ret) + 1):
            train_global = _diff(shifted_data['Global'].iloc[t - win2:t])
            train_region = _diff(region_data.iloc[t - win2:t])

            # Hierarchical DFM factors (global + regional); train/test already split by h
            g_train, g_test, r_train, r_test = hdfm.extract(
                X_global_window=train_global,
                region_cols=list(train_region.columns),
                h=h,
                window_token=(h, train_global.index[-1]),
                region_key=region_key,
            )

            train_global_excl = train_global.drop(columns=data_temp.name, errors='ignore')
            train_ret = ret.iloc[t - win2:t - h]
            test_ret = ret.iloc[t - 1].to_frame().T
            g_test.index = test_ret.index
            r_test.index = test_ret.index

            train_slices = {
                'Ret': train_ret,
                'Dummies': shifted_dummies.iloc[t - win2:t - h],
                'Inflation': train_global_excl.iloc[:-h],
                'Lags': _lags(data_temp.iloc[t - win2:t - h], max_lags).set_index(train_ret.index[max_lags:]),
                'GlobalFactor': g_train,
                'RegionFactor': r_train,
            }
            test_slices = {
                'Ret': test_ret,
                'Dummies': shifted_dummies.iloc[t - 1, :].to_frame().T,
                'Inflation': train_global_excl.iloc[-1].to_frame().T,
                'Lags': _lags(data_temp.iloc[t - win2:t], max_lags).iloc[-1:].set_index(test_ret.index),
                'GlobalFactor': g_test,
                'RegionFactor': r_test,
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
    pd.DataFrame(fc, index=idx, columns=cols).to_excel(OUT_DIR / f"Fcast_RF_DFM_h{h}.xlsx")
    pd.DataFrame(er, index=idx, columns=cols).to_excel(OUT_DIR / f"e_RF_DFM_h{h}.xlsx")
    print(f"h={h}: wrote RF DFM to {OUT_DIR}")