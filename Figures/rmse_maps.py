"""
World maps of country-specific forecasting performance.

Reads the per-country forecast-error workbooks of the RF and RW models
(main 1980-2019 sample) and draws the choropleth world maps in the
paper, one PDF per figure:

    Figure 3        raw RMSE of the RF model, h = 6 (main text)
    Figures 9-11    raw RMSE of the RF model, h = 1, 3, 12
    Figures 12-15   RMSE ratio of the RF vis-a-vis the RW model,
                    h = 1, 3, 6, 12

Raw RMSEs are shown in percentage points (errors x 100); ratios are
the per-country RMSE of RF divided by that of RW. Countries are drawn
on a Robinson projection from the Natural Earth 1:10m admin-0
shapefile (in ne_10m_admin_0_countries/ next to this script);
countries without forecast data are hatched.

Country names in the error workbooks (World Bank style) are matched
to the shapefile's ISO 3166-1 alpha-3 codes via the mapping below.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- settings -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ERRORS_DIR = ROOT / "Replication results" / "Main case"
SHAPEFILE = Path(__file__).resolve().parent / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

HORIZONS = [1, 3, 6, 12]

# the figures: paper figure number -> (kind, horizon)
FIGURES = {
    3: ("RMSE", 6),
    9: ("RMSE", 1),
    10: ("RMSE", 3),
    11: ("RMSE", 12),
    12: ("ratio", 1),
    13: ("ratio", 3),
    14: ("ratio", 6),
    15: ("ratio", 12),
}

# class boundaries of the colour scheme (values above the last bin
# form a final class up to the data maximum)
BINS = {
    "RMSE": [0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    "ratio": [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50],
}

CMAP = matplotlib.colormaps["Reds"].resampled(10)
FIGSIZE = (20, 14)

# error-workbook country name -> shapefile ISO alpha-3 (ADM0_A3)
ISO3 = {
    "Austria": "AUT", "Burundi": "BDI", "Belgium": "BEL",
    "Burkina Faso": "BFA", "Bahamas": "BHS", "Bolivia": "BOL",
    "Brazil": "BRA", "Barbados": "BRB", "Botswana": "BWA",
    "Canada": "CAN", "Switzerland": "CHE", "Chile": "CHL",
    "Cote d'Ivoire": "CIV", "Cameroon": "CMR", "Colombia": "COL",
    "Costa Rica": "CRI", "Cyprus": "CYP", "Germany": "DEU",
    "Dominica": "DMA", "Denmark": "DNK", "Dominican Republic": "DOM",
    "Algeria": "DZA", "Ecuador": "ECU", "Egypt, Arab Rep.": "EGY",
    "Spain": "ESP", "Ethiopia": "ETH", "Finland": "FIN",
    "Fiji": "FJI", "France": "FRA", "United Kingdom": "GBR",
    "Ghana": "GHA", "Gambia, The": "GMB", "Greece": "GRC",
    "Grenada": "GRD", "Guatemala": "GTM", "Honduras": "HND",
    "Haiti": "HTI", "Hungary": "HUN", "Indonesia": "IDN",
    "India": "IND", "Ireland": "IRL", "Iceland": "ISL",
    "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM",
    "Jordan": "JOR", "Japan": "JPN", "Kenya": "KEN",
    "St. Kitts and Nevis": "KNA", "Korea, Rep.": "KOR",
    "St. Lucia": "LCA", "Sri Lanka": "LKA", "Luxembourg": "LUX",
    "Morocco": "MAR", "Madagascar": "MDG", "Mexico": "MEX",
    "Malta": "MLT", "Myanmar": "MMR", "Mauritius": "MUS",
    "Malawi": "MWI", "Malaysia": "MYS", "Niger": "NER",
    "Nigeria": "NGA", "Netherlands": "NLD", "Norway": "NOR",
    "Nepal": "NPL", "Pakistan": "PAK", "Panama": "PAN",
    "Peru": "PER", "Philippines": "PHL", "Portugal": "PRT",
    "Paraguay": "PRY", "Sudan": "SDN", "Senegal": "SEN",
    "Singapore": "SGP", "Solomon Islands": "SLB",
    "El Salvador": "SLV", "Suriname": "SUR", "Slovenia": "SVN",
    "Sweden": "SWE", "Eswatini": "SWZ", "Seychelles": "SYC",
    "Thailand": "THA", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkey": "TUR", "Taiwan, China": "TWN",
    "Uruguay": "URY", "United States": "USA", "Samoa": "WSM",
    "South Africa": "ZAF",
}


def rmse(e):
    """Per-country RMSE, skipping NaNs outside the evaluation window
    (same as Functions.rmse, redefined here so that drawing the maps
    does not require the forecasting libraries Functions.py loads)."""
    e = e[~np.isnan(e)]
    return np.sqrt(np.mean(e ** 2))


def map_values(kind, h):
    """Series of values to plot, indexed by ISO alpha-3 code."""
    rmse_rf = pd.read_excel(ERRORS_DIR / "RF" / f"e_RF_h{h}.xlsx", index_col=0).apply(rmse)
    if kind == "RMSE":
        values = rmse_rf * 100  # percentage points
    else:
        rmse_rw = pd.read_excel(ERRORS_DIR / "RW" / f"e_RW_h{h}.xlsx", index_col=0).apply(rmse)
        values = rmse_rf / rmse_rw
    unmapped = values.index.difference(ISO3)
    if len(unmapped):
        raise KeyError(f"countries missing from the ISO3 mapping: {list(unmapped)}")
    return values.rename(index=ISO3)


def draw_map(gdf, values, bins, outfile):
    merged = gdf.merge(values.rename("value"), how="left",
                       left_on="ADM0_A3", right_index=True)
    ax = merged.dropna(subset=["value"]).plot(
        column="value", cmap=CMAP, figsize=FIGSIZE, scheme="User_Defined",
        legend=True, edgecolor="#aaaaaa", classification_kwds=dict(bins=bins))
    merged[merged["value"].isna()].plot(
        ax=ax, color="#ffffff", hatch="////", edgecolor="#aaaaaa")
    ax.set_axis_off()
    ax.set_xlim([-1.5e7, 1.7e7])
    ax.get_legend().set_bbox_to_anchor((.12, .4))
    ax.get_figure().savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(ax.get_figure())


# --- draw the maps ---------------------------------------------------
if __name__ == "__main__":
    gdf = gpd.read_file(SHAPEFILE)[["ADM0_A3", "geometry"]].to_crs("+proj=robin")
    for fig_no, (kind, h) in FIGURES.items():
        infix = "MapRMSE" if kind == "RMSE" else "MapRMSEratios"
        name = f"Figure{fig_no}_{infix}_RF_h{h}.pdf"
        draw_map(gdf, map_values(kind, h), BINS[kind], RESULTS_DIR / name)
        print(name)
