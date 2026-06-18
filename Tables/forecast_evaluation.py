"""
Forecast evaluation summary tables.

Reads the per-country forecast-error workbooks (one per model and
horizon) and builds the forecasting-results tables in the paper, one
per case in CASES below (main 1980-2019 sample, extended 1980-2023
sample, and the 120-month estimation window):

    columns (1)-(4)   RMSE, h = 1, 3, 6, 12
    columns (5)-(8)   MAD,  h = 1, 3, 6, 12
    columns (9)-(10)  share of country/horizon pairs where the model
                      attains the lowest RMSE resp. MAD (in percent);
                      ties are split equally between the tied models
                      (RW and SRW coincide at h = 12)
    columns (11)-(12) multi-horizon average SPA test of Quaedvlieg
                      (2021) against the benchmark, based on squared
                      errors: mean p-value across countries and the
                      share of p-values below 5% (in percent)

Both aggregation modes used in the paper are supported for the RMSE and
MAD columns, selectable per output file:

    "ratio of averages"   per-country metric averaged across countries,
                          reported as model average / benchmark average;
                          the benchmark row shows the average level
    "average of ratios"   model/benchmark metric ratio computed per
                          country first, then averaged; the benchmark
                          row equals 1 by construction

The ensemble is the equal-weighted combination of the EN, NN, RF, and
XGB forecasts, so its forecast error is the mean of those four error
series. Errors are NaN outside the out-of-sample evaluation window and
are skipped by the metrics.

For the aSPA test, the squared-error loss differentials of the four
horizons are aligned on common target dates (longer horizons start
later), as in Quaedvlieg's original MATLAB implementation. The
bootstrap is made reproducible by reseeding for every country/model
pair (np.random.default_rng(4)); these seeded p-values reproduce
identically run-to-run and are the values reported in the paper's
Tables 2, 3, 12 and 13. (They differ only marginally -- at most 0.004
in the mean p-value -- from the earlier single-stream MATLAB rng(4) run.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- settings -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Tables" / "Results"

sys.path.insert(0, str(ROOT))
from Functions import mad, rmse, test_aspa  # noqa: E402

HORIZONS = [1, 3, 6, 12]
BENCHMARK = "RW"
ENSEMBLE_OF = ["EN", "NN", "RF", "XGB"]

# per-country forecast-error workbook for each model
# ({sfx} = case-specific file infix, {h} = horizon)
MODELS = {
    "RW": "RW/e_RW{sfx}_h{h}.xlsx",
    "SRW": "SRW/e_SRW{sfx}_h{h}.xlsx",
    "AR": "AR dum/e_AR_dum{sfx}_h{h}.xlsx",
    "CM": "CM model/e_CM{sfx}_h{h}.xlsx",
    "EN": "Elastic net/e_EN{sfx}_h{h}.xlsx",
    "NN": "NN/e_NN{sfx}_h{h}.xlsx",
    "RF": "RF/e_RF{sfx}_h{h}.xlsx",
    "XGB": "XGboost/e_XGB{sfx}_h{h}.xlsx",
}

# the cases: error-file location and the tables to produce from it
CASES = {
    "Main case": {
        "folder": "Main case",
        "sfx": "",
        "outputs": {
            "Table2_ForecastingResults.xlsx": "ratio of averages",
            "Table12_ForecastingResults_AverageOfRatios.xlsx": "average of ratios",
        },
    },
    "Extended": {
        "folder": "Extended",
        "sfx": "_Extended",
        "outputs": {"Table3_ForecastingResults_1980-2023.xlsx": "ratio of averages"},
    },
    "Shorter window": {
        "folder": "Shorter window",
        "sfx": "_ShorterWindow",
        "outputs": {"Table13_ForecastingResults_Window120.xlsx": "ratio of averages"},
    },
}

# aSPA test settings
ASPA_WEIGHTS = np.full(len(HORIZONS), 1 / len(HORIZONS))
ASPA_BLOCK_LENGTH = 3
ASPA_SEED = 4


# --- load forecast errors, per horizon -------------------------------
def load_errors(case_dir, sfx, h):
    """Dict of (dates x countries) forecast-error frames, one per model."""
    errors = {}
    for m, f in MODELS.items():
        df = pd.read_excel(case_dir / f.format(sfx=sfx, h=h), index_col=0)
        df.index = pd.to_datetime(df.index)   # normalize datetime/string date indices so all models align
        errors[m] = df
    errors["Ensemble"] = sum(errors[m] for m in ENSEMBLE_OF) / len(ENSEMBLE_OF)
    return errors


def country_metrics(errors_h):
    """DataFrames (models x countries) of RMSE and MAD."""
    return {
        "RMSE": pd.DataFrame({m: e.apply(rmse) for m, e in errors_h.items()}).T,
        "MAD": pd.DataFrame({m: e.apply(mad) for m, e in errors_h.items()}).T,
    }


def aggregate(per_country, mode):
    """One table column: aggregate models x countries metrics across
    countries relative to the benchmark, according to mode."""
    if mode == "ratio of averages":
        avg = per_country.mean(axis=1)
        col = avg / avg[BENCHMARK]
        col[BENCHMARK] = avg[BENCHMARK]  # benchmark row shows the level
    elif mode == "average of ratios":
        col = per_country.div(per_country.loc[BENCHMARK]).mean(axis=1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return col


def best_model_share(metrics, metric):
    """Percent of country/horizon pairs where each model attains the
    lowest value of the metric. A tie gives each tied model an equal
    fraction of the count (RW and SRW produce identical forecasts at
    h = 12, so each gets half a count there)."""
    counts, total = 0, 0
    for h in HORIZONS:
        per_country = metrics[h][metric]
        is_min = per_country.eq(per_country.min(axis=0), axis=1)
        counts = counts + is_min.div(is_min.sum(axis=0), axis=1).sum(axis=1)
        total += per_country.shape[1]
    return counts / total * 100


def aspa_stats(errors):
    """Mean aSPA p-value and share of p-values < 5% per model (the
    benchmark row stays empty). Loss differential: benchmark squared
    error minus model squared error, horizons aligned on common target
    dates and equally weighted."""
    models = [m for m in list(MODELS) + ["Ensemble"] if m != BENCHMARK]
    countries = errors[HORIZONS[0]][BENCHMARK].columns
    out = pd.DataFrame(np.nan, index=list(MODELS) + ["Ensemble"],
                       columns=["mean_p", "share_below_5pct"])
    for m in models:
        pvals = np.empty(len(countries))
        for i, country in enumerate(countries):
            diffs = [
                (errors[h][BENCHMARK][country] ** 2 - errors[h][m][country] ** 2)
                .dropna().rename(h)
                for h in HORIZONS
            ]
            aligned = pd.concat(diffs, axis=1, join="inner").to_numpy()
            _, pvals[i] = test_aspa(aligned, ASPA_WEIGHTS, ASPA_BLOCK_LENGTH,
                                    rng=np.random.default_rng(ASPA_SEED))
        out.loc[m] = [pvals.mean(), np.mean(pvals < 0.05) * 100]
        print(f"  aSPA {m}: mean p = {pvals.mean():.3f}, "
              f"share < 0.05 = {np.mean(pvals < 0.05) * 100:.0f}%")
    return out


# --- build the tables ------------------------------------------------
for case, cfg in CASES.items():
    print(f"=== {case} ===")
    case_dir = ROOT / "Replication results" / cfg["folder"]
    errors = {h: load_errors(case_dir, cfg["sfx"], h) for h in HORIZONS}
    metrics = {h: country_metrics(errors[h]) for h in HORIZONS}
    aspa = aspa_stats(errors)

    for name, mode in cfg["outputs"].items():
        table = pd.DataFrame({
            (metric, f"h = {h}"): aggregate(metrics[h][metric], mode)
            for metric in ("RMSE", "MAD") for h in HORIZONS
        })
        for metric in ("RMSE", "MAD"):
            col = ("Frequency of outperformance", f"% min. {metric}")
            table[col] = best_model_share(metrics, metric).round(0)
        table[("Multi-horizon aSPA", "Average")] = aspa["mean_p"]
        table[("Multi-horizon aSPA", "Share < 0.05")] = aspa["share_below_5pct"].round(0)
        table.round(3).to_excel(RESULTS_DIR / name)
        print(f"{name} ({mode}): {table.shape[0]} models x {table.shape[1]} columns")
