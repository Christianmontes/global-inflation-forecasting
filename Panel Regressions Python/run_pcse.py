#!/usr/bin/env python3
"""
run_pcse.py - Python reproduction of the R/Stata panel regressions (paper Table 4).

Two-way fixed-effects OLS with Beck-Katz pairwise panel-corrected standard errors
(PCSE) and the nmk small-sample correction. Verified identical to the original
R/Stata implementation to ~1e-7 on every coefficient, standard error, t-statistic, N and k.

Reproduces the Stata command:
    xtpcse lossdhH openess D.gdp D.gdp_percapita avginf unemp i.year i.id, pairwise nmk

Inputs:  Data/Panelh{1,3,6,12}.csv   (one panel per forecast horizon h)

For each horizon h it fits:
  - full      : lossdh{h} ~ openess + d_gdp + d_gdp_percapita + avginf + unemp + FE(year) + FE(id)
  - openness  : lossdh{h} ~ openess + FE(year) + FE(id)

The openness-only model is reported on TWO samples:
  - 'orig'  : its own non-missing sample        -> reproduces the originally submitted table
  - 'align' : the full-model sample (controls present) -> the sample-consistent version used
              in the paper. The two differ because the full model first-differences GDP
              (dropping each country's first year) and requires the controls to be present.

Run:  python run_pcse.py        (needs pandas + numpy; see requirements.txt)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "Data")
HORIZONS = [1, 3, 6, 12]
FULL_REGRESSORS = ["openess", "d_gdp", "d_gdp_percapita", "avginf", "unemp"]


def make_diffs(df, id_col, time_col, cols):
    """Panel-wise first differences (NA for each entity's first observation)."""
    df = df.sort_values([id_col, time_col]).copy()
    for c in cols:
        df["d_" + c] = df.groupby(id_col)[c].diff()
    return df


def bk_pcse_vcov(id_vec, time_vec, resid, X):
    """Beck-Katz pairwise PCSE covariance matrix (ported from the original R)."""
    id_vec = np.asarray(id_vec).astype(str)
    time_vec = np.asarray(time_vec).astype(str)
    resid = np.asarray(resid, float)
    X = np.asarray(X, float)
    k = X.shape[1]
    entities = np.array(sorted(set(id_vec)))
    eidx = {e: i for i, e in enumerate(entities)}
    look = {e: {} for e in entities}
    for e, t, r in zip(id_vec, time_vec, resid):
        look[e][t] = r
    Nent = len(entities)
    Sigma = np.zeros((Nent, Nent))
    for i, ei in enumerate(entities):
        di = look[ei]
        if not di:
            continue
        Sigma[i, i] = sum(v * v for v in di.values()) / len(di)
        for j in range(i + 1, Nent):
            dj = look[entities[j]]
            common = set(di) & set(dj)
            if common:
                Sigma[i, j] = Sigma[j, i] = sum(di[t] * dj[t] for t in common) / len(common)
    M = np.zeros((k, k))
    for tt in set(time_vec):
        idx = np.where(time_vec == tt)[0]
        Xt = X[idx, :]
        rows = [eidx[e] for e in id_vec[idx]]
        M += Xt.T @ Sigma[np.ix_(rows, rows)] @ Xt
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ M @ XtX_inv


def twoway_demean(y, idv, tv, tol=1e-12, maxit=5000):
    """Iterative two-way demeaning, for the genuine within-R^2."""
    y = np.asarray(y, float).copy()
    idv, tv = np.asarray(idv), np.asarray(tv)
    for _ in range(maxit):
        y0 = y.copy()
        y = y - pd.Series(y).groupby(idv).transform("mean").values
        y = y - pd.Series(y).groupby(tv).transform("mean").values
        if np.max(np.abs(y - y0)) < tol:
            break
    return y


def fit_pcse(df, dep, regressors, id_col="id", time_col="year"):
    """Two-way FE OLS via dummies + Beck-Katz PCSE (nmk). Returns coef/se/t for `regressors`."""
    d = df.copy()
    d[id_col] = d[id_col].astype(str)
    d[time_col] = d[time_col].astype(str)
    yd = pd.get_dummies(d[time_col], prefix="y", drop_first=True).astype(float)
    idd = pd.get_dummies(d[id_col], prefix="i", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[regressors].astype(float), yd, idd], axis=1)
    Xm, y = X.values, d[dep].astype(float).values
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta
    N, k = Xm.shape
    V = bk_pcse_vcov(d[id_col].values, d[time_col].values, resid, Xm) * (N / (N - k))
    diagV = np.diag(V)
    # Some fixed-effect dummies get a negative PCSE diagonal -> NaN se (benign; R warns identically).
    with np.errstate(invalid="ignore"):
        se = np.where(diagV > 0, np.sqrt(diagV), np.nan)
    cols = list(X.columns)
    out = {r: (beta[cols.index(r)], se[cols.index(r)], beta[cols.index(r)] / se[cols.index(r)])
           for r in regressors}
    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2_overall = 1 - ssr / sst
    yw = twoway_demean(y, d[id_col].values, d[time_col].values)
    r2_within = 1 - ssr / float((yw ** 2).sum())
    return dict(coef=out, N=N, k=k, r2_overall=r2_overall, r2_within=r2_within)


def load(h):
    df = pd.read_csv(os.path.join(DATA, f"Panelh{h}.csv"))
    df.columns = [c.lower() for c in df.columns]
    return make_diffs(df, "id", "year", ["gdp", "gdp_percapita"])


def main():
    print("=" * 78)
    print("Table 4 panel regressions - Python (two-way FE OLS + Beck-Katz PCSE, nmk)")
    print("=" * 78)
    table = {}
    for h in HORIZONS:
        df = load(h)
        dep = f"lossdh{h}"
        full_cols = [dep] + FULL_REGRESSORS + ["id", "year"]
        df_full = df.dropna(subset=full_cols)
        df_open = df.dropna(subset=[dep, "openess", "id", "year"])
        rf = fit_pcse(df_full, dep, FULL_REGRESSORS)
        oa_orig = fit_pcse(df_open, dep, ["openess"])
        oa_align = fit_pcse(df_full, dep, ["openess"])
        table[h] = (rf, oa_orig, oa_align)
        print(f"\n----- h = {h} -----")
        oo, oa = oa_orig["coef"]["openess"], oa_align["coef"]["openess"]
        print(f"  openness-only  orig  (N={oa_orig['N']}): b={oo[0]:.6f}  t={oo[2]:.6f}  R2={oa_orig['r2_overall']:.4f}")
        print(f"  openness-only  align (N={oa_align['N']}): b={oa[0]:.6f}  t={oa[2]:.6f}  R2={oa_align['r2_overall']:.4f}")
        print(f"  full model (N={rf['N']}):")
        for v in FULL_REGRESSORS:
            b, se, t = rf["coef"][v]
            print(f"    {v:<16} b={b:+.6f}  se={se:.6f}  t={t:+.6f}")
        print(f"    overall R2={rf['r2_overall']:.4f}   within R2={rf['r2_within']:.4f}")

    print("\n" + "=" * 78)
    print("CORRECTED TABLE 4  (coef (t) [R^2];  openness-only on the full-model sample)")
    print("=" * 78)
    for h in HORIZONS:
        rf, _, oa = table[h]
        b = lambda v: rf["coef"][v]
        o = oa["coef"]["openess"]
        print(f"h={h:>2}  open-only: {o[0]:+.3f} ({o[2]:+.3f}) [{oa['r2_overall']:.3f}] N={oa['N']}")
        print(f"      full     : "
              f"open {b('openess')[0]:+.3f}({b('openess')[2]:+.3f})  "
              f"dGDP {b('d_gdp')[0]:+.3f}({b('d_gdp')[2]:+.3f})  "
              f"dGDPpc {b('d_gdp_percapita')[0]:+.3f}({b('d_gdp_percapita')[2]:+.3f})  "
              f"unemp {b('unemp')[0]:+.3f}({b('unemp')[2]:+.3f})  "
              f"avginf {b('avginf')[0]:+.3f}({b('avginf')[2]:+.3f})  [{rf['r2_overall']:.3f}] N={rf['N']}")


if __name__ == "__main__":
    main()
