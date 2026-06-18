"""Figure 33: SHAP partial dependence plots for five case-study countries.

For each country and horizon (h = 1 and h = 6) one predictor is shown in a SHAP
dependence plot -- the predictor's value on the x-axis against its SHAP value on
the y-axis -- for a single forecast date (February 2018 for h = 1, February 2009
for h = 6). In each panel the predictor is selected by its importance rank: the
k-th most important predictor by mean absolute SHAP, with k set to reproduce the
predictor shown in the paper's Figure 33 (named in the comment beside each rank).

SHAP values are read from ``RF_shapvalues_train_only`` and the matching feature
values from ``RF_datasets_train_only`` (saved directly by Forecast_RF.py); the
two are aligned row-for-row and column-for-column.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "Replication results" / "Main case" / "RF_shapvalues_train_only"
DATA_DIR = ROOT / "Replication results" / "Main case" / "RF_datasets_train_only"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

COLOR = "#882255"
DOT_SIZE = 25

# One panel per (country, horizon, date, importance rank), in display order.
# The rank is the predictor's position in the mean-absolute-SHAP ranking; the
# comment names the predictor it lands on (matching the paper's Figure 33).
PANELS = [
    ("United States", 1, "2018-02-01", 2),   # Germany
    ("United States", 6, "2009-02-01", 5),   # Mexico
    ("Nigeria", 1, "2018-02-01", 0),         # Cote d'Ivoire
    ("Nigeria", 6, "2009-02-01", 1),         # Global factor
    ("Brazil", 1, "2018-02-01", 0),          # First lag
    ("Brazil", 6, "2009-02-01", 0),          # First lag
    ("Japan", 1, "2018-02-01", 2),           # Korea, Rep.
    ("Japan", 6, "2009-02-01", 1),           # Uruguay
    ("United Kingdom", 1, "2018-02-01", 4),  # Netherlands
    ("United Kingdom", 6, "2009-02-01", 3),  # Global factor
]

# Cosmetic predictor labels for the x-axis.
RENAME = {"lag_0": "First lag", "lag_1": "Second lag", "lag_2": "Third lag",
          "lag_3": "Fourth lag", "MeanGlobal": "Global factor",
          "MeanRegion": "Regional factor"}

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 19))
for ax, (country, h, date, rank) in zip(axes.flatten(), PANELS):
    shapvalues = pd.read_pickle(SHAP_DIR / f"Shapvalues_rf_{country}_h{h}_{date}.sav")
    features = pd.read_pickle(DATA_DIR / f"Dataset_rf_{country}_h{h}_{date}.sav").drop("Returns", axis=1)

    top_inds = np.argsort(-np.mean(np.abs(shapvalues.values), 0))
    ind = int(top_inds[rank])
    features = features.rename(columns=RENAME)

    shap.dependence_plot(ind, shapvalues.values, features, interaction_index=None,
                         color=COLOR, dot_size=DOT_SIZE, ax=ax, show=False)
    ax.set_title(f"{country}: h={h}")
    ax.set_ylabel("")

plt.tight_layout()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(RESULTS_DIR / "Figure33_DependencePlots.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Wrote {RESULTS_DIR / 'Figure33_DependencePlots.pdf'}")
