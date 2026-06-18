"""
Descriptive statistics for the global inflation data (paper Tables 8-10).

    Table 8   full sample        1980-2019
    Table 9   first sub-sample   1980-1999
    Table 10  second sub-sample  2000-2019

Per country we report mean, std, min, max of monthly inflation (in
percent) and the first-order autocorrelation (AC1). Each regional block
ends with a cross-country "Average" row. Just run it top to bottom.

The data sit in one workbook with a two-row header: row 1 = continent,
row 2 = country.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- settings -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Processed data" / "Data_Global.xlsx"
RESULTS_DIR = ROOT / "Tables" / "Results"

SPLIT = 239  # monthly 1980-02 .. 2019-12; rows [:239]=1980-99, [239:]=2000-19

REGION_ORDER = ["Europe", "North America", "South America", "AsiaOceania", "Africa"]
REGION_LABELS = {"AsiaOceania": "Asia and Oceania"}  # display name in the tables

# the three tables: (output name, rows to use)
SAMPLES = {
    "Table8_Descriptives_1980-2019": slice(None),
    "Table9_Descriptives_1980-1999": slice(0, SPLIT),
    "Table10_Descriptives_2000-2019": slice(SPLIT, None),
}

# --- load -----------------------------------------------------------
data = pd.read_excel(DATA_FILE, header=[0, 1], index_col=0)
data.columns = data.columns.set_names(["Continent", "Country"])
continents = data.columns.get_level_values("Continent")

# first-order autocorrelation (biased; matches statsmodels acf(.., nlags=1)[1])
ac1 = lambda x: np.sum((x - x.mean())[1:].values * (x - x.mean())[:-1].values) / np.sum((x - x.mean()) ** 2)

# --- build each table -----------------------------------------------
for name, rows in SAMPLES.items():
    sample = data.iloc[rows]

    blocks = []
    for region in REGION_ORDER:

        countries = sorted(data.columns[continents == region].get_level_values("Country"))
        panel = sample[region][countries]

        stats = pd.DataFrame({
            "Mean": panel.mean() * 100,
            "Std": panel.std(ddof=0) * 100,
            "Min": panel.min() * 100,
            "Max": panel.max() * 100,
            "AC1": panel.apply(ac1),
        })
        stats.loc["Average"] = stats.mean()

        stats.index = pd.MultiIndex.from_product([[REGION_LABELS.get(region, region)], stats.index])
        blocks.append(stats)

    table = pd.concat(blocks)
    table.round(3).to_excel(f"{RESULTS_DIR}/{name}.xlsx")
    print(f"{name}: {len(table)} rows")
