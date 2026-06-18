"""Elastic Net forecast.

Rolling-window, expanding out-of-sample forecasts for every country and horizon
h in {1, 3, 6, 12}, using a 20-year (240-month) window so forecasts start in
2000. Predictors are the seasonal dummies, AR lags, the global and regional mean
factors, and the cross-country inflation panel; they are standardised before the
Elastic Net is tuned (l1_ratio fixed, alpha by cross-validation) and fit. Results
are written to 'Replication results/Extended/Elastic net' as Fcast_EN_*/e_EN_*.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _en_tune, _en_pred, _lags, _diff, load_panels, make_month_dummies

DATA_FILE = ROOT / "Processed data" / "Data_Global_extended.xlsx"
OUT_DIR = ROOT / "Replication results" / "Extended" / "Elastic net"

max_lags = 4
win1 = 12 * 20            # rolling window length (~20 years -> forecasts from 2000)
horizon_vector = [1, 3, 6, 12]
model_config_en = {"l1_ratio": [0.5]}

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

        alpha = l1 = None  # tuned at the first window and each December
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
            if train.empty or test.empty:
                continue

            # Standardise the predictors (fit on train, apply to both)
            X_cols = [c for c in train.columns if c != "Returns"]
            scaler = StandardScaler().fit(train[X_cols])
            train_std = pd.concat([train[["Returns"]],
                                   pd.DataFrame(scaler.transform(train[X_cols]), index=train.index, columns=X_cols)], axis=1)
            test_std = pd.concat([test[["Returns"]],
                                  pd.DataFrame(scaler.transform(test[X_cols]), index=test.index, columns=X_cols)], axis=1)

            if (t == win2) or (pd.to_datetime(test_std.index[0]).month == 12) or (alpha is None):
                alpha, l1 = _en_tune(train_std, 'Returns', h, model_config_en)

            yhat, _ = _en_pred(train_std, test_std, 'Returns', alpha, l1)
            yhat = yhat.iloc[0] if hasattr(yhat, "iloc") else yhat
            fc[t - 1, i] = yhat
            er[t - 1, i] = test_std['Returns'].iloc[0] - yhat

    idx, cols = shifted_data['Global'].index, shifted_data['Global'].columns
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fc, index=idx, columns=cols).to_excel(OUT_DIR / f"Fcast_EN_Extended_h{h}.xlsx")
    pd.DataFrame(er, index=idx, columns=cols).to_excel(OUT_DIR / f"e_EN_Extended_h{h}.xlsx")
    print(f"h={h}: wrote EN to {OUT_DIR}")
