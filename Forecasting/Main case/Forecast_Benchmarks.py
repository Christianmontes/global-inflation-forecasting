"""Benchmark forecasts:
  RW     - random walk (last observed value)
  SRW    - seasonal random walk (value one year earlier)
  AR dum - dummy-augmented AR(p), p selected by BIC
  CM     - Ciccarelli-Mojon: dummy-augmented AR(p) plus the cross-sectional
           global and regional inflation factors

Rolling-window, expanding out-of-sample forecasts for every country and horizon
h in {1, 3, 6, 12}, using a 20-year (240-month) window so forecasts start in
2000. Each model's forecasts and errors are written to its sub-folder of
'Replication results/Main case' as Fcast_*/e_* Excel files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import ar_dummies_forecast, ar_dummies_factor_forecast, _diff, load_panels, make_month_dummies

DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
MAIN_CASE = ROOT / "Replication results" / "Main case"

max_lags = 4
win1 = 12 * 20            # rolling window length (~20 years -> forecasts from 2000)
horizon_vector = [1, 3, 6, 12]

# model key -> (sub-folder, forecast file prefix, error file prefix)
outputs = {
    "RW":    ("RW", "Fcast_RW", "e_RW"),
    "SRW":   ("SRW", "Fcast_SRW", "e_SRW"),
    "ARdum": ("AR dum", "Fcast_AR_dum", "e_AR_dum"),
    "CM":    ("CM model", "Fcast_CM", "e_CM"),
}

data, continent_row = load_panels(DATA_FILE)
dummies = make_month_dummies(data['Global'].index, drop=None)  # keep all 12
ncol = data['Global'].shape[1]

for h in horizon_vector:
    win2 = win1 + h

    # Shift features/dummies by the horizon
    shifted_data = {region: df.shift(periods=h).dropna() for region, df in data.items()}
    shifted_dummies = dummies.shift(periods=h).dropna()
    n_rows = len(shifted_data['Global'])

    fc = {m: np.full((n_rows, ncol), np.nan) for m in outputs}
    er = {m: np.full((n_rows, ncol), np.nan) for m in outputs}

    for i in range(ncol):
        print(f"h={h} country {i}")

        region_key = str(continent_row.iloc[i]).replace(" ", "")
        region_data = shifted_data.get(region_key)

        data_temp = data['Global'].iloc[1:, i]
        ret = data_temp.rolling(h).mean().dropna().to_frame("Returns")

        for t in range(win2, len(ret) + 1):
            train_ret = ret[t - win2:t - h]
            test_ret = ret.iloc[t - 1, :].to_frame().T
            y_true = test_ret['Returns'].iloc[0]

            # Random walk: last observed value
            rw = train_ret['Returns'].iloc[-1]
            fc['RW'][t - 1, i] = rw
            er['RW'][t - 1, i] = y_true - rw

            # Seasonal random walk: value one year before the target
            srw = train_ret['Returns'].iloc[h - 12 - 1]
            fc['SRW'][t - 1, i] = srw
            er['SRW'][t - 1, i] = y_true - srw

            # Dummy-augmented AR(p), p chosen by BIC
            f_ar = ar_dummies_forecast(train_ret, test_ret, shifted_dummies, "Returns",
                                       data_temp[t - win2:t - h], data_temp[t - win2:t], max_lags)
            f_ar = f_ar if np.isscalar(f_ar) else f_ar[0]
            fc['ARdum'][t - 1, i] = f_ar
            er['ARdum'][t - 1, i] = y_true - f_ar

            # Ciccarelli-Mojon: AR(p) + dummies + global/regional mean factors
            train_global = _diff(shifted_data['Global'].iloc[t - win2:t])
            train_region = _diff(region_data.iloc[t - win2:t])
            mean_global = pd.DataFrame(np.mean(train_global, axis=1), columns=["MeanGlobal"], index=train_global.index)
            mean_region = pd.DataFrame(np.mean(train_region, axis=1), columns=["MeanRegion"], index=train_region.index)
            train_factors = pd.concat([mean_global[:-h], mean_region[:-h]], axis=1).dropna()
            test_factors = pd.concat([mean_global.iloc[-1].to_frame().T,
                                      mean_region.iloc[-1].to_frame().T], axis=1).dropna()
            f_cm = ar_dummies_factor_forecast(train_ret, test_ret, shifted_dummies, "Returns",
                                              data_temp[t - win2:t - h], data_temp[t - win2:t],
                                              max_lags, train_factors, test_factors)
            f_cm = f_cm if np.isscalar(f_cm) else f_cm[0]
            fc['CM'][t - 1, i] = f_cm
            er['CM'][t - 1, i] = y_true - f_cm

    # Save forecasts and errors to the matching Main case sub-folders
    idx, cols = shifted_data['Global'].index, shifted_data['Global'].columns
    for m, (sub, fp, ep) in outputs.items():
        (MAIN_CASE / sub).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fc[m], index=idx, columns=cols).to_excel(MAIN_CASE / sub / f"{fp}_h{h}.xlsx")
        pd.DataFrame(er[m], index=idx, columns=cols).to_excel(MAIN_CASE / sub / f"{ep}_h{h}.xlsx")
    print(f"h={h}: wrote RW / SRW / AR dum / CM to {MAIN_CASE}")
