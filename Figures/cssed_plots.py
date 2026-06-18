"""
Cumulative sum of squared error difference (CSSED) figures.

Reads the per-country forecast-error workbooks (main 1980-2019 sample)
and draws the CSSED figures in the paper, one 2 x 2 panel
(h = 1, 3, 6, 12) per figure. The CSSED of a model is the running sum
of the benchmark's squared errors minus the model's, per country, over
the out-of-sample window; a rising line means the model is beating the
RW benchmark. Two styles are drawn for every model:

    percentile curves   the 80th/60th/median/40th/20th percentile of
                        the cross-country CSSED distribution at each
                        point in time (a line follows a point of the
                        distribution, not a country):
                        Figure 2 (RF, main text) and Figures 25-31
                        (remaining models, appendix F.5)
    specific countries  the CSSED path of the five countries located
                        at those percentiles of the end-of-sample
                        CSSED distribution (legend shows the country):
                        Figures 17-24 (appendix F.4)

The SRW figures omit the h = 12 panel: at that horizon the SRW
coincides with the RW benchmark, so the CSSED is identically zero.
NBER recession periods are shaded; recessions that end before a
horizon's out-of-sample window starts are left out (this drops the
2001 recession at h = 12). The ensemble is the equal-weighted
combination of the EN, NN, RF, and XGB forecasts, as in
Tables/forecast_evaluation.py.
"""

from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

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

# the figures: paper figure number -> (model, style)
FIGURES = {
    2: ("RF", "percentiles"),
    17: ("SRW", "countries"),
    18: ("AR", "countries"),
    19: ("CM", "countries"),
    20: ("EN", "countries"),
    21: ("NN", "countries"),
    22: ("RF", "countries"),
    23: ("XGB", "countries"),
    24: ("Ensemble", "countries"),
    25: ("SRW", "percentiles"),
    26: ("AR", "percentiles"),
    27: ("CM", "percentiles"),
    28: ("EN", "percentiles"),
    29: ("NN", "percentiles"),
    30: ("XGB", "percentiles"),
    31: ("Ensemble", "percentiles"),
}

PERCENTILES = [80, 60, 50, 40, 20]
PERCENTILE_LABELS = ["80th percentile", "60th percentile", "Median",
                     "40th percentile", "20th percentile"]
COLORS = ["#882255", "#AA4499", "#999933", "#117733", "#44AA99"]

# NBER recession periods shaded in the figures
NBER_RECESSIONS = [("2001-04-01", "2001-11-01"),
                   ("2007-12-01", "2009-06-01")]


def load_cssed(h):
    """Dict of (dates x countries) CSSED frames over the out-of-sample
    window, one per non-benchmark model."""
    errors = {m: pd.read_excel(ERRORS_DIR / f.format(h=h), index_col=0)
              for m, f in MODELS.items()}
    errors["Ensemble"] = sum(errors[m] for m in ENSEMBLE_OF) / len(ENSEMBLE_OF)
    oos = 239 + h  # errors are NaN during the first 240-month estimation window
    dates = pd.to_datetime(errors[BENCHMARK].index[oos:], format="%m/%d/%Y")
    rw_sq = errors[BENCHMARK].iloc[oos:] ** 2
    return {m: (rw_sq - errors[m].iloc[oos:] ** 2).cumsum().set_axis(dates)
            for m in errors if m != BENCHMARK}


def draw_cssed(cssed_by_h, model, style, outfile):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for ax, h in zip(axes.flatten(), HORIZONS):
        if model == "SRW" and h == 12:  # SRW = RW at h = 12, CSSED is zero
            ax.axis("off")
            continue
        cssed = cssed_by_h[h][model]

        if style == "countries":
            # the countries at the percentiles of the final CSSED distribution
            ranks = cssed.iloc[-1].rank(method="max", pct=True)
            n = len(ranks)
            for p, color in zip(PERCENTILES, COLORS):
                country = ranks.index[ranks == round(p / 100 * n) / n][0]
                ax.plot(cssed.index, cssed[country], label=country, color=color)
        else:
            for p, label, color in zip(PERCENTILES, PERCENTILE_LABELS, COLORS):
                ax.plot(cssed.index, np.percentile(cssed, p, axis=1),
                        label=label, color=color)

        for start, end in NBER_RECESSIONS:
            if pd.to_datetime(end) <= cssed.index[0]:
                continue  # recession over before the out-of-sample window
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                       color="grey", alpha=0.3)

        ax.set_title(f"Forecast horizon: h = {h}")
        if style == "countries":
            ax.legend(framealpha=0.5)
        else:
            ax.legend(loc="upper left", framealpha=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

    fig.tight_layout()
    fig.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)


# --- draw the figures -------------------------------------------------
if __name__ == "__main__":
    cssed_by_h = {h: load_cssed(h) for h in HORIZONS}
    for fig_no, (model, style) in FIGURES.items():
        infix = "_SpecificCountries" if style == "countries" else ""
        name = f"Figure{fig_no}_CSSED_{model}{infix}.pdf"
        draw_cssed(cssed_by_h, model, style, RESULTS_DIR / name)
        print(name)
