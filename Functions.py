from __future__ import annotations

import os
dirname = os.path.dirname(__file__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      

import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import scipy.stats
import math as math
import statsmodels.formula.api as smf
from typing import Dict, Tuple, Optional, Iterable
from dataclasses import dataclass
from pathlib import Path
import pickle
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU
from keras.regularizers import l1
import xgboost as xgb
import optuna
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

def load_panels(
    path: str | Path,
    sheet: int | str = 0
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Read a single Excel sheet structured as:
      Row 0: group/region label for each country column (col0 can be a label)
      Row 1: 'Date' in col0, then country names in col1..end
      Row 2+: data rows: Date, values...

    Returns:
      data: dict with keys 'Global' plus one key per region (spaces removed)
      continent_row: pd.Series mapping each column (country) -> region key
                     aligned to data['Global'].columns order
    """
    path = Path(path)
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError("Workbook too small; expected at least 3 rows and 2 columns.")

    # First row (row 0) holds group names for each country column (skip the Date column at col 0)
    group_row = raw.iloc[0, 1:].astype(str).str.strip()
    group_row_clean = group_row.str.replace(" ", "", regex=False)

    # Second row (row 1) holds the header: Date + country names
    colnames = raw.iloc[1, 1:].astype(str).str.strip().tolist()

    # Data starts at row 2; first column is Date
    dates = pd.to_datetime(raw.iloc[2:, 0], errors="coerce")

    # Numeric block: coerce to numeric, keep NaNs where necessary
    values = raw.iloc[2:, 1:].apply(pd.to_numeric, errors="coerce")

    # Build the unified Global panel
    global_df = pd.DataFrame(values.values, index=dates, columns=colnames)
    global_df = global_df.loc[~global_df.index.isna()].sort_index()
    global_df = global_df.loc[:, global_df.notna().any(axis=0)]

    # Align the group mapping to the final set/order of columns
    continent_row = pd.Series(group_row_clean.values, index=colnames)
    continent_row = continent_row.loc[global_df.columns]

    data: Dict[str, pd.DataFrame] = {"Global": global_df}
    for region in sorted(continent_row.unique()):
        cols = continent_row.index[continent_row == region].tolist()
        if len(cols) == 0:
            continue
        data[region] = global_df[cols].copy()

    assert len(continent_row) == data["Global"].shape[1], "Group row not aligned to columns."

    return data, continent_row


def make_month_dummies(index: pd.Index, drop: str | None = None) -> pd.DataFrame:
    """Create 12 month dummies aligned to the provided DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index, errors="coerce")
    d = pd.get_dummies(index.month, prefix="Month", drop_first=(drop == "last"), dtype=int)
    d.index = index
    return d

def _diff(df):
    """First-difference any inflation series that has an observation more extreme than 50%."""
    Outputdf = pd.DataFrame(np.nan, index = df.index, columns = df.columns )
    
    for columns in df:
        if (0.5 < df[columns]).any() == True:
            diffed = df[columns].diff()
        else:
            diffed = df[columns]

        Outputdf[columns] = diffed
    return Outputdf.bfill()

def _lags(data, lag, raw=False):
   if raw == True:
       lag_range = range(1, lag)
   else:
       lag_range = range(0, lag)

   data = pd.concat([data.shift(i+1) for i in lag_range],axis=1)
   data.columns = ['lag_%d' %i for i in lag_range]

   return data.dropna()


def _saveModel(model, date, modelstring, country_name, h):
          
    date = pd.to_datetime(date)
    stringdate = date.strftime('%Y-%m-%d')
    export_file = country_name + '_h' + str(h) + '_' + stringdate + '.sav'
    pickle.dump(model, open(modelstring + '_' + export_file, 'wb'))

def LoadModel(date, modelstring, country_name, h):
    
    date = pd.to_datetime(date)
    stringdate = date.strftime('%Y-%m-%d')
    load_file = country_name + '_h' + str(h) + '_' + stringdate + '.sav'
    loaded_model = pickle.load(open(modelstring + load_file, 'rb'))
    
    return loaded_model


def _pca(window, name="Factor", n_components=1):
    """Principal-component factor(s) of a rolling-window panel.
    Standardises the panel columns and fits PCA on the whole window, returning
    the first n_components scores as a DataFrame aligned to window.index. Used to
    build the global/regional factors for the PCA-factor forecasts (the analogue
    of the cross-sectional mean factors)."""
    scaled = StandardScaler().fit_transform(window)
    scores = PCA(n_components=n_components).fit_transform(scaled)
    cols = [name] if n_components == 1 else [f"{name}{i + 1}" for i in range(n_components)]

    return pd.DataFrame(scores, columns=cols, index=window.index)

def _Autoencoder_Fit_2Layer_TrainTestCombined(data, n_components, regu, epochs, opt):
    input_X = Input(shape=(data.shape[1],))
    e = Dense(np.sqrt(data.shape[1]), kernel_regularizer=l1(regu))(input_X)
    e = LeakyReLU(alpha=0.3)(e)

    bottleneck = Dense(n_components, kernel_regularizer=l1(regu))(e)
    bottleneck = LeakyReLU(alpha=0.3)(bottleneck)

    d = Dense(np.sqrt(data.shape[1]), activation='relu')(bottleneck)
    d = Dense(data.shape[1], activation='linear')(d)
        
    model = Model(input_X, d)
    model.compile(optimizer = opt, loss ='mse')
    encoder = Model(input_X, bottleneck)
    model.fit(data, data, epochs = epochs, batch_size = 32, verbose=0)

    return encoder.predict(data, verbose=0)


warnings.simplefilter("ignore", ConvergenceWarning)  # keep logs clean; not a fallback

EPS = 1e-12

def _zscore_on_training(df: pd.DataFrame, train_idx: pd.Index) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Column-wise standardization using TRAIN rows only.
    Returns standardized ALL rows + (mu, sd). Zero s.d. columns become NaN and are filled to avoid division-by-zero.
    """
    mu = df.loc[train_idx].mean(axis=0)
    sd = df.loc[train_idx].std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (df - mu) / sd
    return z.fillna(0.0), mu, sd.fillna(1.0)


def _drop_near_constant_cols(df: pd.DataFrame, train_idx: pd.Index, tol: float = 1e-10) -> pd.DataFrame:
    """Drop columns with (train) variance below tol. Returns (possibly) narrower df."""
    sd = df.loc[train_idx].std(axis=0, ddof=0)
    keep = (sd > tol) & np.isfinite(sd)
    dropped = df.columns[~keep]
    if len(dropped) > 0:
        warnings.warn(
            f"Dropping {len(dropped)} near-constant column(s) on train: {list(dropped)[:5]}{'...' if len(dropped)>5 else ''}",
            RuntimeWarning,
        )
    return df.loc[:, keep.index[keep]]


def _first_pc_svd(X: pd.DataFrame, name: str = "Factor") -> pd.Series:
    """
    First principal component using SVD (no sklearn dependency).
    Assumes caller passed data already in the desired scale (e.g., z-scored).
    """
    Xv = X.values
    if not np.isfinite(Xv).all():
        raise ValueError("PCA received non-finite values.")
    U, s, Vt = np.linalg.svd(Xv, full_matrices=False)
    pc1 = U[:, 0] * s[0]
    return pd.Series(pc1, index=X.index, name=name)


def _ensure_factor_series(f_arr, index: pd.Index, name: str) -> pd.Series:
    """Normalize shapes from statsmodels to a pd.Series aligned to `index`."""
    if isinstance(f_arr, pd.DataFrame):
        if f_arr.shape[1] == 1:
            f = f_arr.iloc[:, 0]
        elif f_arr.shape[0] == 1:
            f = f_arr.iloc[0, :].T
        else:
            f = f_arr.stack().iloc[:len(index)]
    elif isinstance(f_arr, pd.Series):
        f = f_arr
    else:
        f = pd.Series(np.asarray(f_arr).squeeze(), index=index)
    if len(f) != len(index):
        f = pd.Series(f.values[:len(index)], index=index)
    return f.rename(name)


def _fit_dfm_first_factor_raw(
    X: pd.DataFrame,
    factor_order: int,
    error_cov_type: str,
    maxiter: int,
    em_iter: int,
    use_smoothed: bool,
    name: str
) -> pd.Series:
    """Single attempt at DFM first factor (no fallbacks)."""
    mod = DynamicFactor(X, k_factors=1, factor_order=factor_order, error_cov_type=error_cov_type)
    res = mod.fit(disp=False, maxiter=maxiter, em_iter=em_iter)

    if use_smoothed and hasattr(res, "factors") and hasattr(res.factors, "smoothed"):
        f_arr = res.factors.smoothed
    elif hasattr(res, "factors") and hasattr(res.factors, "filtered"):
        f_arr = res.factors.filtered
    else:
        f_arr = res.smoothed_state[0]

    return _ensure_factor_series(f_arr, X.index, name)


def _is_good_factor(f: pd.Series, train_idx: pd.Index) -> bool:
    arr = f.loc[train_idx].values
    return np.isfinite(arr).all() and np.nanvar(arr) > EPS


def _fit_first_factor_robust(
    X: pd.DataFrame,
    train_idx: pd.Index,
    *,
    factor_order: int = 1,
    error_cov_type: str = "diagonal",
    maxiter: int = 500,
    em_iter: int = 25,
    use_smoothed: bool = True,
    name: str = "Factor",
    allow_scalar_cov_retry: bool = True,
) -> pd.Series:
    """
    Try DFM → (optional retry with scalar error cov) → PCA fallback.
    Guarantees: finite factor with non-zero variance on train slice.
    """
    # 1) Primary attempt
    try:
        f = _fit_dfm_first_factor_raw(X, factor_order, error_cov_type, maxiter, em_iter, use_smoothed, name)
        if _is_good_factor(f, train_idx):
            return f
    except Exception as e:
        last_err = e
    else:
        last_err = RuntimeError("DFM factor degenerate on train (zero variance or non-finite).")

    # 2) Retry with scalar error covariance (often stabilizes SSM solver)
    if allow_scalar_cov_retry:
        try:
            f = _fit_dfm_first_factor_raw(X, factor_order, "scalar", maxiter, em_iter, use_smoothed, name)
            if _is_good_factor(f, train_idx):
                warnings.warn("DFM recovered with error_cov_type='scalar' after initial failure.", RuntimeWarning)
                return f
        except Exception:
            pass

    # 3) PCA fallback
    warnings.warn(
        f"DFM fit failed or degenerate ({type(last_err).__name__}: {last_err}). Falling back to PCA for {name}.",
        RuntimeWarning,
    )
    f = _first_pc_svd(X, name=name)
    if not _is_good_factor(f, train_idx):
        raise RuntimeError(f"PCA fallback produced a degenerate {name} on train; check inputs (variance/NaNs).")
    return f


def _zscore_factor_train_test(f: pd.Series, h: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize factor using train stats ([:-h]) and return (train_df, test_df) 1-col DataFrames."""
    name = f.name or "Factor"
    if len(f) < h + 1:
        raise ValueError("Factor length shorter than h+1; check window setup.")
    train_idx = f.index[:-h]
    mu = f.loc[train_idx].mean()
    sd = f.loc[train_idx].std(ddof=0)
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    fz = (f - mu) / sd
    return fz.loc[train_idx].to_frame(name), fz.iloc[[-1]].to_frame(name)


def _ols_deflate_by_factor(Xz: pd.DataFrame, f: pd.Series, train_idx: pd.Index) -> pd.DataFrame:
    """
    Deflate standardized Xz by OLS on factor f using TRAIN rows only.
    Vectorized; returns residuals for ALL rows.
    """
    f_train = f.loc[train_idx].values  # (T_train,)
    denom = float(np.dot(f_train, f_train))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Degenerate global factor in train window (zero variance).")
    num = (Xz.loc[train_idx].mul(f_train, axis=0)).sum(axis=0).values  # (N,)
    lambdas = num / denom  # (N,)
    F = f.values.reshape(-1, 1)      # (T,1)
    L = lambdas.reshape(1, -1)       # (1,N)
    resid = Xz.values - (F @ L)      # (T,N)
    return pd.DataFrame(resid, index=Xz.index, columns=Xz.columns)


@dataclass
class HierarchicalDFMExtractor:
    """
    Global → deflate → regional (robust, no caching).
      1) Fit global factor on TRAIN-standardized full panel.
      2) Deflate entire panel by train-OLS loadings on global factor.
      3) TRAIN-standardize residual panel (recommended).
      4) Fit regional factor on residuals restricted to region cols.
      5) Return train/test 1-col DataFrames for both factors (train-only scaling).
    """
    factor_order: int = 1
    error_cov_type: str = "diagonal"
    use_smoothed: bool = True
    maxiter: int = 500
    em_iter: int = 25
    zscore_residuals: bool = True          # re-standardize residuals before regional DFM (recommended)
    prune_near_constant: bool = True       # auto-drop near-constant cols on train (stability)

    def _fit_global(self, X: pd.DataFrame, h: int) -> Tuple[pd.Series, pd.DataFrame]:
        if X.isna().any(axis=None):
            raise ValueError("X contains NaNs; align/diff/fill upstream before calling extractor.")

        train_idx = X.index[:-h]

        # Optionally prune and z-score using train stats
        X_use = _drop_near_constant_cols(X, train_idx) if self.prune_near_constant else X
        Xz, _, _ = _zscore_on_training(X_use, train_idx)

        # Robust global factor
        f = _fit_first_factor_robust(
            Xz, train_idx,
            factor_order=self.factor_order,
            error_cov_type=self.error_cov_type,
            maxiter=self.maxiter,
            em_iter=self.em_iter,
            use_smoothed=self.use_smoothed,
            name="GlobalFactor",
            allow_scalar_cov_retry=True,
        )
        return f, Xz

    def _fit_region_on_residuals(self, resid: pd.DataFrame, region_cols: Iterable[str], h: int) -> pd.Series:
        idx = resid.index
        train_idx = idx[:-h]

        cols_req = list(region_cols) if region_cols else []
        cols = [c for c in cols_req if c in resid.columns]
        if len(cols) == 0:
            # predictable output when region missing
            return pd.Series(np.nan, index=idx, name="RegionFactor")

        Xr = resid.loc[:, cols]

        # Optional pruning + train z-scoring
        if self.prune_near_constant:
            Xr = _drop_near_constant_cols(Xr, train_idx)
        if self.zscore_residuals:
            Xr, _, _ = _zscore_on_training(Xr, train_idx)

        if Xr.shape[1] == 0:
            return pd.Series(np.nan, index=idx, name="RegionFactor")
        if Xr.isna().any(axis=None):
            raise ValueError("Regional residual window contains NaNs after preprocessing; check alignment/differencing.")

        # Robust regional factor
        fr = _fit_first_factor_robust(
            Xr, train_idx,
            factor_order=self.factor_order,
            error_cov_type=self.error_cov_type,
            maxiter=self.maxiter,
            em_iter=self.em_iter,
            use_smoothed=self.use_smoothed,
            name="RegionFactor",
            allow_scalar_cov_retry=True,
        )
        return fr

    def extract(
            self,
            X_global_window: pd.DataFrame,
            region_cols: Iterable[str],
            h: int,
            window_token: Optional[Tuple[int, pd.Timestamp]] = None,
            region_key: Optional[str] = None  # optional cache key for regional factor
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # --- Minimal caching setup (lazy) ---
        if not hasattr(self, "_global_cache"):
            self._global_cache = {}
            self._resid_cache = {}
            self._region_cache = {}

        if window_token is None:
            window_token = (h, X_global_window.index[-1])

        # --- 1) Global fit & residual panel (with cache) ---
        if (window_token in self._global_cache) and (window_token in self._resid_cache):
            f_global = self._global_cache[window_token]
            resid = self._resid_cache[window_token]
        else:
            f_global, Xz = self._fit_global(X_global_window, h)
            train_idx = X_global_window.index[:-h]
            resid = _ols_deflate_by_factor(Xz, f_global, train_idx=train_idx)
            self._global_cache[window_token] = f_global
            self._resid_cache[window_token] = resid

        # --- 2) Regional factor (with cache) ---
        rkey = (window_token[0], window_token[1], (region_key or "REGION"))
        if rkey in self._region_cache:
            f_region = self._region_cache[rkey]
        else:
            f_region = self._fit_region_on_residuals(resid, region_cols, h)
            self._region_cache[rkey] = f_region

        # --- 3) Train/test scaling (train-only stats) ---
        g_train, g_test = _zscore_factor_train_test(f_global, h)
        r_train, r_test = _zscore_factor_train_test(f_region, h)

        return g_train, g_test, r_train, r_test


def BIC(residuals, h):
    """Compute the BIC used to select the optimal AR lag length for the benchmark models."""
    T = len(residuals)
    sum_squared_errors = sum(residuals**2)
    BIC = T*math.log(sum_squared_errors/T) + h*math.log(T)

    return BIC


def ar_dummies_forecast(train, test, dummies, y_key, train_AR_data, test_AR_data, max_lags):
    """Dummy-augmented AR(p) forecast: selects the AR lag order p (1..max_lags) by BIC
    using the AR lags and seasonal dummies, then forecasts one step ahead."""
    bic = np.empty(max_lags) * np.nan

    dataS_train = _lags(train_AR_data, max_lags)
    dataS_train = dataS_train.set_index(train.index[max_lags:])
    
    dummy_matrix = dummies.drop(columns=['Month_12'])
    
    model = LinearRegression()    

    for i in range(1, max_lags+1):
        tempt = pd.concat([dataS_train.iloc[:, :i], dummy_matrix], axis=1).dropna()
        model.fit(tempt, train[y_key][max_lags:])
        pred_fit = model.predict(tempt)
        resid = np.subtract(train[y_key][max_lags:], pred_fit)
        bic[i-1] = BIC(resid, i) 
    
    optimal_lag = int(np.argmin(bic)) + 1
    
    dataS_test = _lags(test_AR_data, optimal_lag)
    dataS_test = dataS_test.iloc[-1:]
    dataS_test = dataS_test.set_index(test.index)

    data_test = pd.concat([dataS_test, dummy_matrix], axis=1).dropna()
    tempt_opt = pd.concat([dataS_train.iloc[:, :optimal_lag], dummy_matrix], axis=1).dropna()   
    model.fit(tempt_opt, train[y_key][max_lags:])

    return model.predict(data_test[-1:])


def ar_dummies_factor_forecast(train, test, dummies, y_key, train_AR_data, test_AR_data, max_lags, train_factors, test_factors):
    """Ciccarelli-Mojon (CM) forecast: a dummy-augmented AR(p) model augmented with the
    cross-sectional global and regional inflation factors. The AR lag order is chosen
    by BIC from the AR lags and seasonal dummies only; the factors enter only the
    final prediction model."""
    bic = np.empty(max_lags) * np.nan
    
    # Create lagged data using "lags" function
    dataS_train = _lags(train_AR_data, max_lags)
    dataS_train = dataS_train.set_index(train.index[max_lags:])
    
    dummy_matrix = dummies.drop(columns=['Month_12'])
    
    model = LinearRegression()    

    for i in range(1, max_lags+1):
        tempt = pd.concat([dataS_train.iloc[:, :i], dummy_matrix], axis=1).dropna()
        model.fit(tempt, train[y_key][max_lags:])
        pred_fit = model.predict(tempt)
        resid = np.subtract(train[y_key][max_lags:], pred_fit)
        bic[i-1] = BIC(resid, i)
    
    optimal_lag = int(np.argmin(bic)) + 1

    dataS_test = _lags(test_AR_data, optimal_lag)
    dataS_test = dataS_test.iloc[-1:]
    dataS_test = dataS_test.set_index(test.index)

    data_test_combined = pd.concat([dataS_test, test_factors, dummy_matrix], axis=1).dropna()
    data_train_combined = pd.concat([dataS_train.iloc[:, :optimal_lag], train_factors, dummy_matrix], axis=1).dropna()
    model.fit(data_train_combined, train[y_key][max_lags:])    

    return model.predict(data_test_combined)


def _en_tune(train, y_key, h, model_config):
    rs = np.random.RandomState(seed = 666)
    ts_cv = TimeSeriesSplit(n_splits=20, gap=h, test_size=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = ElasticNetCV(l1_ratio=model_config["l1_ratio"], fit_intercept=True, n_jobs=20, cv=ts_cv, random_state=rs)
        model.fit(train.drop(y_key, axis=1), train[y_key])
        best_model_l1 = model.l1_ratio_
        best_model_alpha = model.alpha_
    
    return best_model_alpha, best_model_l1


def _en_pred(train, test, y_key, best_model_alpha, best_model_l1):
    rs = np.random.RandomState(seed = 666)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mean_model = ElasticNet(fit_intercept=True, random_state=rs, alpha = best_model_alpha, l1_ratio = best_model_l1).fit(train.drop(y_key, axis=1), train[y_key])
        y_hat_mean = pd.Series(mean_model.predict(test.drop(y_key, axis=1)), index=test.index).rename("en_tune_mean_y_hat")
        preds = y_hat_mean

    return preds, mean_model


def _rfTune(train, y_key, h, model_config):
    model = RandomForestRegressor(
        n_estimators=model_config["n_estimators"],
        n_jobs = 30,
        min_samples_leaf=5,
        random_state=666)

    cv_strategy = TimeSeriesSplit(n_splits=20, gap=h, test_size=1)

    model = GridSearchCV(
        estimator=model,
        param_grid=model_config["grid"],
        cv=cv_strategy,
        verbose=0,
        scoring="neg_mean_squared_error",
        refit=False,
        n_jobs = 30
    )

    model.fit(train.drop(y_key, axis=1), train[y_key])

    return model.best_params_

    
def _rfPred(train, test, y_key, best_model_mean, model_config):
    rs = np.random.RandomState(seed=666)
    model = RandomForestRegressor(
        n_jobs=30,
        n_estimators=model_config["n_estimators"],
        random_state=rs,
        min_samples_leaf=5,
        **best_model_mean
    )
    X_tr, y_tr = train.drop(columns=[y_key]), train[y_key]
    X_te = test.drop(columns=[y_key])
    model.fit(X_tr, y_tr)
    y_hat = pd.Series(model.predict(X_te), index=test.index, name="rf_tune_mean_y_hat")
    return float(y_hat.iloc[0]), model


def _rfOLSPred(train, test, y_key, best_model_mean, model_config):
    rs = np.random.RandomState(seed=666)
    mean_model = RandomForestRegressor(n_jobs=30, n_estimators=model_config["n_estimators"], random_state=rs, **best_model_mean, min_samples_leaf=5, max_leaf_nodes=20)
    mean_model.fit(train.drop(y_key, axis=1), train[y_key])
    ols_model = LinearRegression() 
    preds = []
    
    for tree in mean_model.estimators_:
        features = pd.DataFrame(tree.feature_importances_.reshape((-1,1)).T, columns=train.drop("Returns", axis=1).columns)
        features = features.loc[:, (features != 0).any(axis=0)]
        New_X = train.drop("Returns", axis=1)[features.columns]
        ols_model.fit(New_X, train["Returns"])
        preds.append(ols_model.predict(test.drop(y_key, axis=1)[features.columns]))
     
    return np.mean(preds)


def _xgbTuneOptuna(train, y_key, h, n_trials=100):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 350),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 100, log=True),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_jobs': 30,
            'random_state': 666,
        }

        model = xgb.XGBRegressor(**param)

        cv_strategy = TimeSeriesSplit(n_splits=20, gap=h, test_size=1)
        scores = []

        for train_idx, val_idx in cv_strategy.split(train):
            X_train, X_val = train.drop(y_key, axis=1).iloc[train_idx], train.drop(y_key, axis=1).iloc[val_idx]
            y_train, y_val = train[y_key].iloc[train_idx], train[y_key].iloc[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = np.mean((preds - y_val) ** 2)*1000  # MSE
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=666))
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def _xgbPred(train, test, y_key, best_model_params):
    rs = np.random.RandomState(seed=666)
    model = xgb.XGBRegressor(
        n_jobs=30,
        random_state=rs,
        n_estimators=best_model_params.get('n_estimators', 300),  # Default to 300 if not found
        max_depth=best_model_params.get('max_depth', 6),
        learning_rate=best_model_params.get('learning_rate', 0.1),
        subsample=best_model_params.get('subsample', 1.0),
        colsample_bytree=best_model_params.get('colsample_bytree', 1.0),
        gamma=best_model_params.get('gamma', 0),
        min_child_weight=best_model_params.get('min_child_weight', 1),
        reg_alpha=best_model_params.get('reg_alpha', 0),
        reg_lambda=best_model_params.get('reg_lambda', 1)
    )
    model.fit(train.drop(y_key, axis=1), train[y_key])
    y_hat = pd.Series(model.predict(test.drop(y_key, axis=1)), index=test.index, name="xgb_tune_mean_y_hat")
    return y_hat.iloc[0], model


def _nnpred_parallel(train, test, y_key, h, depth_type, model_config, random_state):
    rs = np.random.RandomState(seed = random_state)
    scaler = StandardScaler().fit(train.drop(y_key, axis=1))
    layer_sizes = [max(1, int(np.round((train.shape[1] - 1) ** x))) for x in
                       model_config["%s_layer_exponents" % depth_type]]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = MLPRegressor(hidden_layer_sizes=layer_sizes, batch_size=model_config["batch_size"],
                             learning_rate="constant", learning_rate_init=model_config["learning_rate"],
                             activation=model_config["activation"], solver="adam",
                             max_iter=model_config["epochs"], n_iter_no_change=model_config["epochs"],
                             verbose=False, random_state=rs, alpha = model_config["alpha"])
              
        model_out = model.fit(scaler.transform(train.drop(y_key, axis=1)), train[y_key])
        pred_out = model.predict(scaler.transform(test.drop(y_key, axis=1)))

    return pred_out, model_out


def mad(e_1):
    e_1 = e_1[~np.isnan(e_1)]
    mad = np.median(np.abs(e_1 - np.median(e_1)))
    return mad


def rmse(e):
    e = e[~np.isnan(e)]
    rmse = np.sqrt(np.mean(e**2))
    return rmse


def _EncompassingTest(e_Raw, e_Factor, h):
    """Encompassing test of Harvey, Leybourne, and Newbold (1998) with Newey-West std errors (Bartlett kernel, maxlags = h).
    Returns the combination weight (lambda) and p-value of the null lambda = 0 for e_Raw and for e_Factor."""
    maxLag = h

    e_Raw = e_Raw[~np.isnan(e_Raw)]
    e_Factor = e_Factor[~np.isnan(e_Factor)]

    d1 = (e_Factor - e_Raw)*e_Factor
    d2 = (e_Raw - e_Factor)*e_Raw

    df1 = pd.DataFrame({'y':d1})
    df2 = pd.DataFrame({'y':d2})
    reg1 = smf.ols('y~1', data=df1).fit(cov_type='HAC',cov_kwds={'maxlags':maxLag,'kernel':'bartlett'})
    reg2 = smf.ols('y~1', data=df2).fit(cov_type='HAC',cov_kwds={'maxlags':maxLag,'kernel':'bartlett'})
    
    P_h = np.shape(e_Raw)[0]
    
    HLN1 = reg1.tvalues.iloc[0]
    HLN2 = reg2.tvalues.iloc[0]
    MHLN1 = HLN1 * ((P_h+1-2*h+(h*(h-1)/P_h))/P_h)
    MHLN2 = HLN2 * ((P_h+1-2*h+(h*(h-1)/P_h))/P_h)
    
    Lambda1 = np.sum(d1)/sum((e_Factor - e_Raw)**2)
    Lambda2 = np.sum(d2)/sum((e_Factor - e_Raw)**2)
    
    pValue1 = scipy.stats.t.sf(abs(MHLN1), df=(P_h-1))
    pValue2 = scipy.stats.t.sf(abs(MHLN2), df=(P_h-1))
    
    return Lambda1, pValue1, Lambda2, pValue2


def qs_weights(x):
    """Quadratic Spectral kernel weights for HAC estimation."""
    x = np.asarray(x)
    arg_qs = 6 * np.pi * x / 5

    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = 3.0 / (arg_qs**2)
        w2 = (np.sin(arg_qs) / arg_qs) - np.cos(arg_qs)
        w_qs = w1 * w2

    return np.where(x == 0, 1.0, w_qs)


def qs_variance(y):
    """Columnwise Quadratic Spectral HAC estimator of the variance."""
    T, N = y.shape
    bw = 1.3 * T ** (1 / 5)

    lags = np.arange(1, T)
    weights = qs_weights(lags / bw)

    omega = np.zeros(N)
    for i in range(N):
        workdata = y[:, i] - np.mean(y[:, i])
        omega[i] = np.dot(workdata, workdata) / T
        for j in range(T - 1):
            omega[i] += 2 * weights[j] * np.dot(workdata[: T - j - 1], workdata[j + 1:]) / T

    return omega


def mbb_variance(y, L):
    """'Natural variance' estimator of moving-block-resampled data."""
    T, N = y.shape
    omega = np.zeros(N)
    ydem = y - np.mean(y, axis=0)
    K = T // L

    for n in range(N):
        temp = ydem[: K * L, n].reshape(K, L, order="F")
        omega[n] = np.mean(np.sum(temp, axis=0) ** 2) / L

    return omega


def get_mbb_id(T, L, rng=None):
    """Indices of one moving-block-bootstrap resample (block length L)."""
    if rng is None:
        rng = np.random.default_rng()

    idx = np.zeros(T, dtype=np.int64)
    idx[0] = rng.integers(1, T + 1)

    for t in range(1, T):
        if (t + 1) % L == 0:
            idx[t] = rng.integers(1, T + 1)
        else:
            idx[t] = idx[t - 1] + 1

        if idx[t] > T:
            idx[t] = 1

    return idx - 1


def bootstrap_aspa(loss_diff, weights, L, B=999, rng=None):
    """Bootstrap algorithm for the aSPA test: returns the test statistic and
    the B bootstrapped statistics."""
    if rng is None:
        rng = np.random.default_rng()

    T = loss_diff.shape[0]

    weighted_loss_diff = np.sum(weights * loss_diff, axis=1, keepdims=True)
    d_ij = np.mean(weighted_loss_diff)
    t_aSPA = np.sqrt(T) * d_ij / np.sqrt(qs_variance(weighted_loss_diff)[0])

    t_aSPA_b = np.zeros(B)
    demeaned_loss_diff = weighted_loss_diff - d_ij
    for b in range(B):
        idx = get_mbb_id(T, L, rng)
        b_lossdiff = demeaned_loss_diff[idx, :]
        zeta_b = mbb_variance(b_lossdiff, L)
        t_aSPA_b[b] = np.sqrt(T) * np.mean(b_lossdiff) / np.sqrt(zeta_b[0])

    return t_aSPA, t_aSPA_b


def test_aspa(loss_diff, weights, L, B=999, rng=None):
    """Average SPA test of Quaedvlieg (2021); Python port of the original MATLAB
    code (Test_aSPA.m and its helpers).

    loss_diff is (T x H) with one column per horizon, weights is length H, and L
    is the bootstrap block length. The loss-differential convention is benchmark
    loss minus model loss, so positive values favor the model. Pass a seeded numpy
    Generator as rng for reproducible p-values. Returns the test statistic and the
    p-value."""
    t_aSPA, t_aSPA_b = bootstrap_aspa(loss_diff, weights, L, B, rng)
    p_value = np.mean(t_aSPA < t_aSPA_b)

    return t_aSPA, p_value

def _BarFigureImportance(shap_values, Variablelist, Average=True):
    """Aggregate mean absolute SHAP values into predictor groups for the
    stacked importance figures. Variablelist holds one group label per column
    of shap_values (first column of the DataFrame). With Average=True each
    group total is divided by the number of predictors in the group."""
    GroupLabels = {"Africa": "AF", "AsiaOceania": "AS+OC", "Europe": "EU",
                   "North America": "NA", "South America": "SA",
                   "Dummy": "Dum", "Lag": "Lags",
                   "Global": "GF", "Regional": "RF"}

    MeanAbsShap = pd.Series(np.nanmean(np.abs(shap_values), axis=0),
                            index=Variablelist.iloc[:, 0].to_numpy())

    Grouped = MeanAbsShap.groupby(level=0)
    Results = Grouped.mean() if Average else Grouped.sum()

    Results = Results.reindex(GroupLabels).fillna(0).rename(GroupLabels)

    return Results

