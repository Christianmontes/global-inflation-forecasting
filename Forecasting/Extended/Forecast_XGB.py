"""XGBoost forecast.

Same features as Forecast_RF.py (seasonal dummies, cross-country inflation panel,
AR lags, global and regional mean factors), but the model is XGBoost. Hyper-
parameters are tuned by Optuna at the first window and re-tuned each December;
the tuned model is then fit and used to forecast.

Rolling-window, expanding out-of-sample forecasts for every country and horizon
h in {1, 3, 6, 12}, using a 20-year (240-month) window so forecasts start in
2000. Results are written to 'Replication results/Extended/XGboost' as
Fcast_XGB_h{h}.xlsx / e_XGB_h{h}.xlsx.

NOTE: Optuna and XGBoost are seeded (see _xgbTuneOptuna / _xgbPred) so the run is
reproducible, but the tuned hyper-parameters still differ from the paper (the
original run was unseeded / a different package version).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _diff, _lags, _xgbTuneOptuna, _xgbPred, load_panels, make_month_dummies

DATA_FILE = ROOT / "Processed data" / "Data_Global_extended.xlsx"
OUT_DIR = ROOT / "Replication results" / "Extended" / "XGboost"

max_lags = 4
win1 = 12 * 20            # rolling window length (~20 years -> forecasts from 2000)
horizon_vector = [1, 3, 6, 12]

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

        best_params = None  # tuned at the first window and each December
        for t in range(win2, len(ret) + 1):
            train_global = _diff(shifted_data['Global'].iloc[t - win2:t])
            train_region = _diff(region_data.iloc[t - win2:t])
            mean_global = pd.DataFrame(np.mean(train_global, axis=1), columns=["MeanGlobal"], index=train_global.index)
            mean_region = pd.DataFrame(np.mean(train_region, axis=1), columns=["MeanRegion"], index=train_region.index)
            train_global_excl = train_global.drop(columns=data_temp.name, errors='ignore')

            train_ret = ret.iloc[t - win2:t - h]
            test_ret = ret.iloc[t - 1].to_frame().T

            train_slices = {
                'Ret': train_ret,
                'Dummies': shifted_dummies.iloc[t - win2:t - h],
                'Inflation': train_global_excl.iloc[:-h],
                'Lags': _lags(data_temp.iloc[t - win2:t - h], max_lags).set_index(train_ret.index[max_lags:]),
                'GlobalFactor': mean_global.iloc[:-h],
                'RegionFactor': mean_region.iloc[:-h],
            }
            test_slices = {
                'Ret': test_ret,
                'Dummies': shifted_dummies.iloc[t - 1, :].to_frame().T,
                'Inflation': train_global_excl.iloc[-1].to_frame().T,
                'Lags': _lags(data_temp.iloc[t - win2:t], max_lags).iloc[-1:].set_index(test_ret.index),
                'GlobalFactor': mean_global.iloc[-1].to_frame().T,
                'RegionFactor': mean_region.iloc[-1].to_frame().T,
            }

            train = pd.concat(train_slices.values(), axis=1).dropna()
            test = pd.concat(test_slices.values(), axis=1).dropna()

            if (t == win2) or (pd.to_datetime(test_ret.index[0]).month == 12) or (best_params is None):
                best_params = _xgbTuneOptuna(train, 'Returns', h)

            yhat_val, _ = _xgbPred(train, test, 'Returns', best_params)
            fc[t - 1, i] = yhat_val
            er[t - 1, i] = test['Returns'].iloc[0] - yhat_val

    idx, cols = shifted_data['Global'].index, shifted_data['Global'].columns
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fc, index=idx, columns=cols).to_excel(OUT_DIR / f"Fcast_XGB_Extended_h{h}.xlsx")
    pd.DataFrame(er, index=idx, columns=cols).to_excel(OUT_DIR / f"e_XGB_Extended_h{h}.xlsx")
    print(f"h={h}: wrote XGB to {OUT_DIR}")