"""Extract autoencoder (AE) factors for the RF_AE forecasts -- all horizons.

For each rolling window and horizon h in {1, 3, 6, 12}, a 2-layer autoencoder is
fit to each (standardised, first-differenced) panel and its first component is
kept as that panel's factor: one global factor plus one factor per region
(Africa, AsiaOceania, Europe, NorthAmerica, SouthAmerica). The six factors are
saved per window so Forecast_RF_AE.py can load them as the global/regional
factors (column 'Global' + one column per region key).

Saved to 'Replication results/Main case/AE_factors' as AutoFactors_All_h{h}_{date}.sav
(date = the window's last shifted date), matching how Forecast_RF_AE.py loads them.

NOTE: the autoencoder is trained with Keras and is NOT deterministic across runs
(random weight initialisation / backend nondeterminism) and differs across TF
versions, so the AE factors -- and hence the RF_AE forecasts -- are not
bit-reproducible. Requires the optional tensorflow/keras extras (requirements.txt).
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # deterministic TF ops (paired with the seed below)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _Autoencoder_Fit_2Layer_TrainTestCombined, _diff, _saveModel, load_panels

DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
AE_DIR = ROOT / "Replication results" / "Main case" / "AE_factors"
AE_DIR.mkdir(parents=True, exist_ok=True)

n_components = 1          # one factor per panel
win1 = 12 * 20           # ~20-year rolling window (forecasts from 2000)
horizon_vector = [1, 3, 6, 12]

# Reproducibility: seed Python/NumPy/TF so the AE factors are stable run-to-run.
# (This does NOT reproduce the paper's AE factors -- those used a different
# seed/TF version, and Keras training is not bit-reproducible across versions.)
tf.keras.utils.set_random_seed(666)

# Autoencoder config
regu = 0.001
epochs = 700

data, continent_row = load_panels(DATA_FILE)
# 'Global' first, then the five region keys; this column order is what
# Forecast_RF_AE.py relies on (it selects by name: 'Global' and the region key).
REGIONS = ["Global", "Africa", "AsiaOceania", "Europe", "NorthAmerica", "SouthAmerica"]
scaler = StandardScaler()

for h in horizon_vector:
    win2 = win1 + h
    shifted = {r: data[r].shift(periods=h).dropna() for r in REGIONS}

    # loop length from the global panel (same convention as the RF scripts)
    ret = data['Global'].iloc[1:, 0].rolling(h).mean().dropna()

    for t in range(win2, len(ret) + 1):
        print(f"h={h} t={t}", flush=True)

        factor_cols = []
        for r in REGIONS:
            window = shifted[r].iloc[t - win2:t]
            diffed = _diff(window)
            scaled = scaler.fit_transform(diffed)
            # fresh optimiser per fit (avoids reusing optimiser state across models)
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
            f = _Autoencoder_Fit_2Layer_TrainTestCombined(scaled, n_components, regu, epochs, opt)
            factor_cols.append(pd.DataFrame(np.asarray(f).reshape(-1, 1), columns=[r], index=window.index))

        factors_all = pd.concat(factor_cols, axis=1).dropna()
        _saveModel(factors_all, factors_all.index[-1], str(AE_DIR / "AutoFactors"), "All", h)

    print(f"h={h}: saved AE factors to {AE_DIR}")