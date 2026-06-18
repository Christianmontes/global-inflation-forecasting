"""Estimate the footprint of the RF non-replication (the max_features near-tie)
on a SUBSAMPLE, since re-running RF for the full panel is infeasible.

Re-runs Forecast_RF.py's exact logic for a sample of countries over the first
DATES out-of-sample months at h=1 (starting at the true first window, so the
tuning schedule matches production), compares each forecast to the stored
'Main case/RF' value, and reports:
  * the share of country-month forecasts that differ (|diff| > DIFF_TOL),
  * the magnitude of those differences,
  * how many of the sampled countries are affected,
  * each country's RMSE over the window, fresh vs stored (the quantity that
    drives the reported RMSE ratios).
Extrapolate the share to the full panel (91 countries x ~238 OOS months x 4 h).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Functions import _diff, _lags, _rfTune, _rfPred, load_panels, make_month_dummies

DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
REF_DIR = ROOT / "Replication results" / "Main case" / "RF"

DATES = 60                     # first 60 OOS months (~5 years, spans ~5 Decembers)
max_lags = 4
win1 = 12 * 20
horizon_vector = [1]
model_config_rf = {"n_estimators": 500, "grid": {"max_features": [None, 2 / 3, 1 / 3, "sqrt", "log2"]}}
WANT = ["United States", "Nigeria", "Brazil", "Japan", "United Kingdom", "Germany",
        "India", "Mexico", "South Africa", "Indonesia"]
DIFF_TOL = 1e-6                # a max_features flip moves the forecast by ~1e-3; exact match ~1e-15

data, continent_row = load_panels(DATA_FILE)
cols_all = list(data['Global'].columns)
SUBSET = [c for c in WANT if c in cols_all]
dummies = make_month_dummies(data['Global'].index, drop=None)
ncol = data['Global'].shape[1]
col_idx = {c: data['Global'].columns.get_loc(c) for c in SUBSET}


def run_rf():
    fc = {h: np.full((len(data['Global'].shift(h).dropna()), ncol), np.nan) for h in horizon_vector}
    er = {h: np.full_like(fc[h], np.nan) for h in horizon_vector}
    for h in horizon_vector:
        win2 = win1 + h
        shifted_data = {region: df.shift(periods=h).dropna() for region, df in data.items()}
        shifted_dummies = dummies.shift(periods=h).dropna()
        for c in SUBSET:
            i = col_idx[c]
            print(f"[RF] h={h} {c}", flush=True)
            region_data = shifted_data.get(str(continent_row.iloc[i]).replace(" ", ""))
            data_temp = data['Global'].iloc[1:, i]
            ret = data_temp.rolling(h).mean().dropna().to_frame("Returns")

            best_params = None  # tuned at the first window (and each December)
            for t in range(win2, min(win2 + DATES, len(ret) + 1)):
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
                    best_params = _rfTune(train, 'Returns', h, model_config_rf)

                yhat_val, _ = _rfPred(train, test, 'Returns', best_params, model_config_rf)
                fc[h][t - 1, i] = yhat_val
                er[h][t - 1, i] = test['Returns'].iloc[0] - yhat_val
    return fc, er


def report(fc, er):
    cols = data['Global'].columns
    h = 1
    ref = pd.read_excel(REF_DIR / f"Fcast_RF_h{h}.xlsx", index_col=0)[cols]
    eref = pd.read_excel(REF_DIR / f"e_RF_h{h}.xlsx", index_col=0)[cols]
    got = pd.DataFrame(fc[h], index=range(len(fc[h])), columns=cols)
    eg = pd.DataFrame(er[h], index=range(len(er[h])), columns=cols)

    tot, tot_diff, mags, affected = 0, 0, [], 0
    print("\n per-country (h=1, first %d OOS months):" % DATES)
    for c in SUBSET:
        m = got[c].notna().to_numpy()
        d = np.abs(got[c].to_numpy()[m] - ref[c].to_numpy()[m])
        n, ndiff = int(m.sum()), int((d > DIFF_TOL).sum())
        tot += n; tot_diff += ndiff; mags += d[d > DIFF_TOL].tolist(); affected += (ndiff > 0)
        ef, es = eg[c].to_numpy()[m], eref[c].to_numpy()[m]
        ratio = np.sqrt(np.nanmean(ef ** 2)) / np.sqrt(np.nanmean(es ** 2))
        print(f"   {c:<16} n={n:3d}  differ={ndiff:3d} ({100*ndiff/max(n,1):3.0f}%)   RMSE fresh/stored={ratio:.5f}")
    mags = np.array(mags)
    print("\n================ RF FOOTPRINT ESTIMATE (h=1) ================")
    print(f"  sampled cells          : {tot}")
    print(f"  cells differing        : {tot_diff}  ({100*tot_diff/max(tot,1):.1f}%)")
    print(f"  countries affected     : {affected} / {len(SUBSET)}")
    if mags.size:
        print(f"  |diff| on those cells  : median={np.median(mags):.2e}  max={mags.max():.2e}")
    print("=============================================================")


if __name__ == "__main__":
    print(f"RF footprint estimate: {len(SUBSET)} countries x first {DATES} OOS months, h=1\n", flush=True)
    fc, er = run_rf()
    report(fc, er)