"""
Stacked variable-importance figures for the RF model.

Draws the stacked bar charts of group-level variable importance in the paper,
one 2 x 2 panel (h = 1, 3, 6, 12) per figure:

    Figure 4    total importance of each predictor group (main text)
    Figure 32   average importance per predictor in each group (appendix F.6)

Predictors are grouped into the five continents, seasonal dummies, AR lags, and
the global and regional factors. Within each bar the group shares are scaled to
sum to one.

The group shares are computed directly from the raw per-country SHAP files in
``RF_shapvalues_train_only`` rather than from pre-aggregated region files. For
every country/horizon we read all rows of every rolling-window file, take the
absolute SHAP values and rescale each row to sum to one, and average them per
predictor (ignoring the country's own missing column). Those per-predictor means
are pooled across the countries of each target region (and globally) and then
aggregated into the predictor groups. The variable list (one group label per
predictor) is reconstructed from the SHAP columns and the country -> continent
mapping in the Data folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# --- settings -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "Replication results" / "Main case" / "RF_shapvalues_train_only"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

sys.path.insert(0, str(ROOT))
from Functions import _BarFigureImportance  # noqa: E402

HORIZONS = [1, 3, 6, 12]

# Country -> continent mapping for grouping the predictor columns.
raw_countries = pd.read_excel(ROOT / "Data" / "Raw data.xlsx", nrows=0).columns.drop("Date")
continents = pd.read_excel(ROOT / "Data" / "Continentlist.xlsx", header=None).iloc[0]
country_to_continent = dict(zip(raw_countries, continents))


def _variable_type(col):
    if col.startswith("Month_"):
        return "Dummy"
    if col.startswith("lag_"):
        return "Lag"
    if col == "MeanGlobal":
        return "Global"
    if col == "MeanRegion":
        return "Regional"
    return country_to_continent[col]


# Target regions: (name, continent filter or None for global, x position label).
REGIONS = [("Global", None, "GL"), ("Africa", "Africa", "AF"),
           ("AsiaOceania", "AsiaOceania", "AS & OC"), ("Europe", "Europe", "EU"),
           ("North America", "North America", "NA"), ("South America", "South America", "SA")]


def _country_stats(country, h):
    """Sum and count of row-standardized absolute SHAP per predictor, pooled
    over all rows of all rolling-window files for one country/horizon."""
    total = count = None
    for f in SHAP_DIR.glob(f"Shapvalues_rf_{country}_h{h}_*.sav"):
        a = pd.read_pickle(f).abs()
        a = a.div(a.sum(axis=1), axis="rows")  # rescale each row to sum to one
        s, c = a.sum(axis=0), a.notna().sum(axis=0)
        total = s if total is None else total.add(s, fill_value=0)
        count = c if count is None else count.add(c, fill_value=0)
    return total, count


# Pool per-predictor means by region (each file read once per horizon).
region_mean = {}
for h in HORIZONS:
    stats = {c: _country_stats(c, h) for c in raw_countries}
    for name, continent, _ in REGIONS:
        members = raw_countries if continent is None else \
            [c for c in raw_countries if country_to_continent[c] == continent]
        total = pd.concat([stats[c][0] for c in members], axis=1).sum(axis=1)
        count = pd.concat([stats[c][1] for c in members], axis=1).sum(axis=1)
        region_mean[(h, name)] = total / count
    print(f"h={h}: pooled SHAP for {len(REGIONS)} regions")

# --- draw the figures -----------------------------------------------
width = 0.5
Colors = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
          "#DDCC77", "#CC6677", "#882255", "#AA4499"]
Groups = ["AF", "AS+OC", "EU", "NA", "SA", "Dum", "Lags", "GF", "RF"]


def _stack_bar(ax, Results, x):
    bottom = 0
    for group, color in zip(Groups, Colors):
        ax.bar(height=Results[group], bottom=bottom, x=x, width=width, color=color)
        bottom = bottom + Results[group]


# Average = False reproduces Figure 4, Average = True reproduces Figure 32
for Average, outfile in [(False, "Figure4_StackedImportance.pdf"),
                         (True, "Figure32_StackedImportance_Average.pdf")]:

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, h in zip(axes, HORIZONS):
        for x, (name, _, _) in enumerate(REGIONS):
            mean = region_mean[(h, name)]
            Variablelist = pd.DataFrame([_variable_type(c) for c in mean.index], columns=["Type"])

            Results = _BarFigureImportance(mean.to_numpy().reshape(1, -1), Variablelist, Average=Average)
            Results = Results / np.sum(Results)

            _stack_bar(ax, Results, x)

        ax.set_title("h=" + str(h))
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels([r[2] for r in REGIONS])

    plt.tight_layout()

    leg = fig.legend(
        labels=["Africa", "Asia & Oceania", "Europe", "North America",
                "South America", "Seasonal dummies", "AR",
                "Global factor", "Regional factor"],
        loc="lower center", borderaxespad=0, ncol=3, bbox_to_anchor=(0.5, -0.08),
    )
    for handle, color in zip(leg.legend_handles, Colors):
        handle.set_color(color)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {RESULTS_DIR / outfile}")
