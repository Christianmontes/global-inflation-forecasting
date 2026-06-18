"""Neural-network (MLP ensemble) forecast.

Same features as Forecast_RF.py (seasonal dummies, cross-country inflation panel,
AR lags, global and regional mean factors), but the model is a deep multi-layer
perceptron. For each window an ensemble of NN_MODELS MLPs is fit and the median
prediction is used.

Rolling-window, expanding out-of-sample forecasts for every country and horizon
h in {1, 3, 6, 12}, using a 10-year (120-month) window (after dropping the first
10 years of the sample) so forecasts start in 2000. Results are written to
'Replication results/Shorter window/NN' as
Fcast_NN_h{h}.xlsx / e_NN_h{h}.xlsx.

NOTE: each ensemble member is seeded (random_state = member index) so the run is
reproducible; this differs from the paper, whose ensemble was unseeded.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _diff, _lags, _nnpred_parallel, load_panels, make_month_dummies

DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
OUT_DIR = ROOT / "Replication results" / "Shorter window" / "NN"

max_lags = 4
win1 = 12 * 10            # rolling window length (10 years; see initial_cutoff below)
horizon_vector = [1, 3, 6, 12]
NN_MODELS = 51            # ensemble size; median prediction is used

model_config_nn = {
    "batch_size": 32,
    "learning_rate": 0.01,
    "epochs": 700,
    "activation": "relu",
    "shallow_layer_exponents": [1 / 2],
    "deep_layer_exponents": [3 / 4, 2 / 4, 1 / 4],
    "alpha": 0.01,
}

data, continent_row = load_panels(DATA_FILE)

# Shorter-window sample: drop the first 10 years, then use a 10-year rolling window
initial_cutoff = 12 * 10
data = {region: df.iloc[initial_cutoff:].copy() for region, df in data.items()}
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

            # Ensemble of MLPs (seeded per member); median prediction
            results = Parallel(n_jobs=18)(
                delayed(_nnpred_parallel)(train, test, 'Returns', h, 'deep', model_config_nn, random_state=j)
                for j in range(NN_MODELS)
            )
            y_hat_med = float(np.median([np.ravel(x[0])[0] for x in results]))

            fc[t - 1, i] = y_hat_med
            er[t - 1, i] = test['Returns'].iloc[0] - y_hat_med

    idx, cols = shifted_data['Global'].index, shifted_data['Global'].columns
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fc, index=idx, columns=cols).to_excel(OUT_DIR / f"Fcast_NN_ShorterWindow_h{h}.xlsx")
    pd.DataFrame(er, index=idx, columns=cols).to_excel(OUT_DIR / f"e_NN_ShorterWindow_h{h}.xlsx")
    print(f"h={h}: wrote NN to {OUT_DIR}")