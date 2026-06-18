"""
Box plots of country-specific forecasting performance.

Reads the per-country forecast-error workbooks (main 1980-2019 sample)
and draws the box-plot figures in the paper, one 2 x 2 panel
(h = 1, 3, 6, 12) per figure:

    Figure 1    raw RMSE distributions, all models incl. the RW
                benchmark (main text)
    Figure 16   RMSE ratio distributions relative to RW, all models
                excl. the benchmark (appendix F.3)

Each box spans the interquartile range with whiskers at 1.5 x IQR;
values beyond the whiskers are not shown. The green triangle marks the
mean, the orange line the median. The ensemble is the equal-weighted
combination of the EN, NN, RF, and XGB forecasts, as in
Tables/forecast_evaluation.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- settings -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ERRORS_DIR = ROOT / "Replication results" / "Main case"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

HORIZONS = [1, 3, 6, 12]
BENCHMARK = "RW"
ENSEMBLE_OF = ["EN", "NN", "RF", "XGB"]

# per-country forecast-error workbook for each model ({h} = horizon)
MODELS = {
    "RW": "RW/e_RW_h{h}.xlsx",
    "SRW": "SRW/e_SRW_h{h}.xlsx",
    "AR": "AR dum/e_AR_dum_h{h}.xlsx",
    "CM": "CM model/e_CM_h{h}.xlsx",
    "EN": "Elastic net/e_EN_h{h}.xlsx",
    "NN": "NN/e_NN_h{h}.xlsx",
    "RF": "RF/e_RF_h{h}.xlsx",
    "XGB": "XGboost/e_XGB_h{h}.xlsx",
}

# the figures: output file -> (kind, models shown, in order)
FIGURES = {
    "Figure1_BoxplotRMSE.pdf": (
        "RMSE", ["RW", "SRW", "AR", "CM", "EN", "NN", "RF", "XGB", "Ensemble"]),
    "Figure16_BoxplotRMSEratios.pdf": (
        "ratio", ["SRW", "AR", "CM", "EN", "NN", "RF", "XGB", "Ensemble"]),
}


def rmse(e):
    """Per-country RMSE, skipping NaNs outside the evaluation window."""
    e = e[~np.isnan(e)]
    return np.sqrt(np.mean(e ** 2))


def country_rmse(h):
    """DataFrame (countries x models) of per-country RMSEs."""
    errors = {m: pd.read_excel(ERRORS_DIR / f.format(h=h), index_col=0)
              for m, f in MODELS.items()}
    errors["Ensemble"] = sum(errors[m] for m in ENSEMBLE_OF) / len(ENSEMBLE_OF)
    return pd.DataFrame({m: e.apply(rmse) for m, e in errors.items()})


def draw_boxplots(rmse_by_h, kind, models, outfile):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for i, (ax, h) in enumerate(zip(axes.flatten(), HORIZONS)):
        data = rmse_by_h[h]
        if kind == "ratio":
            data = data.div(data[BENCHMARK], axis=0)
        data = data[models]

        ax.boxplot(data, showfliers=False, showmeans=True)
        ax.set_title(f"Forecast horizon: h = {h}")
        ax.yaxis.grid(True, linestyle="-", which="major",
                      color="lightgrey", alpha=0.5)

        if i >= len(HORIZONS) - 2:  # bottom row only
            ax.set_xticklabels(models, rotation=45)
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(outfile, format="pdf")
    plt.close(fig)


# --- draw the figures -------------------------------------------------
if __name__ == "__main__":
    rmse_by_h = {h: country_rmse(h) for h in HORIZONS}
    for name, (kind, models) in FIGURES.items():
        draw_boxplots(rmse_by_h, kind, models, RESULTS_DIR / name)
        print(name)
