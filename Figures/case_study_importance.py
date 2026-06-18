"""Figure 5: variable importance by predictor group for five case-study
countries (US, Nigeria, Brazil, Japan, UK) at horizons h = 1 and h = 6.

Reads the per-country RF SHAP files, stacks every row, scales each row to sum
to one, aggregates predictors into nine groups (five continents, seasonal
dummies, AR lags, global and regional factors), and draws a 5 x 2 bar panel.
"""

import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "Replication results" / "Main case" / "RF_shapvalues_train_only"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

sys.path.insert(0, str(ROOT))
from Functions import _BarFigureImportance  # noqa: E402

COUNTRIES = ["United States", "Nigeria", "Brazil", "Japan", "United Kingdom"]
HORIZONS = [1, 6]
LABELS = ["AF", "AS+OC", "EU", "NA", "SA", "Seasonality", "AR", "GL factor", "RE factor"]
COLORS = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
          "#DDCC77", "#CC6677", "#882255", "#AA4499"]

# Map each predictor column to its group (countries -> continent via the Data folder).
raw = pd.read_excel(ROOT / "Data" / "Raw data.xlsx", nrows=0).columns.drop("Date")
continent = dict(zip(raw, pd.read_excel(ROOT / "Data" / "Continentlist.xlsx", header=None).iloc[0]))
group = lambda c: {"Month_": "Dummy", "lag_": "Lag"}.get(c[:c.find("_") + 1]) or \
    {"MeanGlobal": "Global", "MeanRegion": "Regional"}.get(c) or continent[c]

fig, axes = plt.subplots(len(COUNTRIES), len(HORIZONS), figsize=(16, 19), sharex=True)
for row, country in enumerate(COUNTRIES):
    for col, h in enumerate(HORIZONS):
        shap = pd.concat(map(pd.read_pickle, SHAP_DIR.glob(f"Shapvalues_rf_{country}_h{h}_*.sav"))).abs()
        shap = shap.div(shap.sum(axis=1), axis="rows")
        types = pd.DataFrame([group(c) for c in shap.columns], columns=["Type"])
        shares = _BarFigureImportance(shap.to_numpy(), types, Average=False)
        shares /= shares.sum()
        axes[row, col].bar(LABELS, shares.to_numpy(), color=COLORS)
        axes[row, col].set_title(f"{country}: h={h}")

for ax in axes[-1, :]:
    ax.tick_params(axis="x", rotation=50, labelsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "Figure5_CaseStudies.pdf", format="pdf", bbox_inches="tight")
