# Reproducibility package — *Global Inflation Forecasting*

International Journal of Forecasting, manuscript IJF-D-25-00472R1.

- **Package assembled:** 2026-06-19
- **Authors of the paper:** Marcelo C. Medeiros, Erik Christian Montes Schutte, and Tobias Skipper Soussi
- **Package authors / contacts for the reproducibility check:** Tobias Skipper Soussi — tss@econ.au.dk; Erik Christian Montes Schutte — christianms@econ.au.dk
- **License:** MIT — see [`LICENSE`](LICENSE).
- **Code authorship:** all code in this repository was written by **Tobias Skipper Soussi** and **Erik Christian Montes Schutte**. Claude Code (an AI assistant) was used only to help organize, document, and package the repository for the reproducibility check; it did not write any of the code.

---

## 1. What this package does

The paper produces, for a panel of monthly CPI inflation series:

1. rolling-window, expanding out-of-sample **forecasts** for many models
   (random walk, seasonal random walk, dummy-augmented AR, Ciccarelli–Mojon,
   Elastic Net, Random Forest and its PCA/DFM/autoencoder/OLS variants, a
   neural-net ensemble, XGBoost, and an equal-weight ensemble), across three
   samples (main 1980–2019, extended 1980–2023, and a 120-month-window variant);
2. the **tables and figures** in the paper, computed from the per-country
   forecast-error files.

There are **two ways** to reproduce the results (details in §6):

- **Path A — tables & figures only (fast, exact).** Run the scripts in `Tables/`
  and `Figures/` on the forecast-error files already provided in
  `Replication results/`. This reproduces every table and figure in the paper
  **exactly** (verified bit-for-bit). The multi-horizon aSPA columns of
  Tables 2, 3, 12 and 13 come from a *seeded* moving-block bootstrap
  (`np.random.default_rng(4)`): they reproduce identically run-to-run, and the
  printed tables report exactly these values. See the note in
  `Tables/forecast_evaluation.py`.
- **Path B — regenerate the forecasts (very heavy).** Re-run the scripts in
  `Forecasting/` to recompute the forecast-error files, then run Path A. Most
  models reproduce to machine precision; a few do not reproduce *bit-for-bit*,
  but the differences are tiny and do not change
  any conclusion in the paper.

---

## 2. Repository structure

```
.
├── README.md                     this file
├── requirements.txt              Python env for Forecasting/ and Tables/ (pinned)
├── requirements-figures.txt      Python env for Figures/ (separate; see §5)
├── Functions.py                  shared functions used by all Python scripts
│
├── Data/                         RAW inputs
│   ├── Raw data.xlsx                 raw CPI data, main sample
│   ├── Raw data extended.xlsx        raw CPI data, extended sample
│   ├── Continentlist.xlsx            country → region mapping (main)
│   └── Continentlist extended.xlsx   country → region mapping (extended)
│
├── Data work/                    MATLAB data-construction scripts (see §3, §4)
│   ├── build_data_global.m
│   ├── factors_em.m
│   └── prepare_missing.m
│
├── Processed data/               INTERMEDIARY panels (built by Data work/*.m)
│   ├── Data_Global.xlsx          main sample, input to Forecasting/
│   └── Data_Global_extended.xlsx extended sample
│
├── Forecasting/                  PATH B — regenerates the forecast-error files
│   ├── Main case/                main sample (20-yr window, forecasts from 2000)
│   ├── Extended/                 extended sample (1980–2023)
│   └── Shorter window/           10-yr window variant
│
├── Replication results/          INTERMEDIARY forecast outputs (errors + forecasts)
│   ├── Main case/                one sub-folder per model (RW, SRW, AR dum, CM
│   │                             model, Elastic net, RF, RF_PCA, RF_DFM, RF_AE,
│   │                             RFOLS, NN, XGboost, Encompassing) + AE_factors,
│   │                             RF_shapvalues_train_only, RF_datasets_train_only
│   │                             (last two are large, ~16.5 GB; SHAP figures — see §3)
│   ├── Extended/
│   └── Shorter window/
│
├── Tables/                       PATH A — builds the paper tables
│   ├── *.py
│   └── Results/                  the table workbooks (Table2 … Table13)
│
├── Figures/                      PATH A — builds the paper figures
│   ├── *.py
│   ├── ne_10m_admin_0_countries/ Natural Earth 1:10m shapefile (for the maps; public domain)
│   └── Results/                  the figure PDFs (Figure1 … Figure33)
│
└── Panel Regressions Python/     Table 4 panel regressions (run_pcse.py + Data/Panelh{1,3,6,12}.csv)
```

---

## 3. Data

- **Format:** Excel workbooks (`.xlsx`). The `Data_Global*` panels have a two-row
  header (row 1 = region/continent, row 2 = country; first column = date) and
  monthly observations thereafter.
- **Raw data** (`Data/`): monthly headline CPI inflation series from the World
  Bank's **Global Database of Inflation** (Prospects Group), plus a
  country→region mapping. Source page:
  <https://www.worldbank.org/en/research/brief/inflation-database>. Please cite
  Ha, J., M. A. Kose, and F. Ohnsorge (2023), "One-Stop Source: A Global Database
  of Inflation," *Journal of International Money and Finance* 137: 102896. The
  database is publicly downloadable (Excel / Stata, free) and is updated roughly
  twice a year; no explicit license is stated on the source page.
  - **Vintages / revisions (important for reproduction).** The World Bank does
    **not** publish dated vintages of this database, and it occasionally
    **revises historical values back in time**. The exact series underlying the
    paper therefore cannot be guaranteed to be re-downloadable from the source
    as-is — a fresh download may differ slightly from ours. For this reason the
    **raw vintage we used is bundled in this package** (`Data/Raw data.xlsx` and
    `Data/Raw data extended.xlsx`), and the reproduction runs from those included
    files rather than from a new download.
- **Intermediary data** (`Processed data/`): `Data_Global.xlsx` and
  `Data_Global_extended.xlsx` are produced from the raw data by the
  MATLAB scripts in `Data work/` (`build_data_global.m`, which uses
  `prepare_missing.m` and `factors_em.m` for the EM-based handling of missing
  observations). They are included so the forecasting code runs directly; they
  can be regenerated from the raw data with MATLAB.
- **Panel-regression inputs** (`Panel Regressions Python/Data/Panelh{1,3,6,12}.csv`, one file
  per horizon): the country–year panel behind **Table 4** (determinants of forecasting
  outperformance), for 86 countries over 2000–2019. The dependent variable is the annual
  RW−RF squared-loss differential (derived from this package's RW and RF forecast errors in
  `Replication results/Main case/`); the regressors are trade openness ((imports + exports) /
  GDP), GDP growth, GDP-per-capita growth, the unemployment rate, and the average inflation
  rate. Inflation is from the Global Database of Inflation (above); the macro regressors are
  from the **World Bank World Development Indicators**
  (WDI, <https://databank.worldbank.org/source/world-development-indicators>; free, public,
  no registration or cost) — imports of goods and services (% of GDP, `NE.IMP.GNFS.ZS`),
  exports of goods and services (% of GDP, `NE.EXP.GNFS.ZS`), GDP growth (annual %,
  `NY.GDP.MKTP.KD.ZG`), GDP-per-capita growth (annual %, `NY.GDP.PCAP.KD.ZG`), and
  unemployment (% of total labour force, `SL.UEM.TOTL.ZS`), **downloaded 23 August 2022**.
  (WDI is periodically
  revised, so a fresh download may differ; the panel is therefore shipped as the intermediary
  file below.) These CSVs are **intermediary
  data**, shipped directly so `Panel Regressions Python/run_pcse.py` reproduces Table 4 without
  rebuilding them (the upstream assembly scripts are not included — per the IJF allowance for
  shipping intermediary files).
- **Map shapefile** (`Figures/ne_10m_admin_0_countries/`): Natural Earth 1:10m
  Admin-0 countries, used only by `rmse_maps.py`. Natural Earth data is in the
  **public domain**; the files are bundled for convenience.
- **Intermediary forecast outputs** (`Replication results/`): per-country
  forecast and forecast-error workbooks, one per model and horizon — the output
  of `Forecasting/` (Path B) and the input to `Tables/`/`Figures/` (Path A).
  Also here: `AE_factors/` (autoencoder factors),
  `RF_shapvalues_train_only/` and `RF_datasets_train_only/` (per-window SHAP
  values and training frames used only by the SHAP figures).
  - **Large SHAP intermediaries — included in the repository.**
    `RF_shapvalues_train_only/` (~16 GB, 83,356 files) and
    `RF_datasets_train_only/` (~0.5 GB, 2,330 files) are the per-window SHAP
    values and training frames used **only** by the four SHAP figures
    (4, 5, 32, 33). They are **included in this repository** so those figures
    reproduce directly from Path A; they can also be regenerated by
    `Forecasting/Main case/Forecast_RF.py`, which writes them as it runs.

### Data lineage at a glance (raw → intermediary → outputs)

Every file marked *intermediary* below is **included** in the package so the check runs without
the heavy upstream steps; the "built by" column names the code that produces it (per the IJF
guidance to flag intermediary data and the code that generates it).

| Pipeline | Raw inputs (bundled) | Intermediary (bundled) | Built by | Used to produce |
|---|---|---|---|---|
| **Forecasting** | `Data/Raw data*.xlsx` (CPI inflation) | `Processed data/Data_Global*.xlsx` | `Data work/*.m` — MATLAB (`build_data_global.m`, `factors_em.m`, `prepare_missing.m`) | `Replication results/` forecast-error files → Tables 2, 3, 5–13 and the figures |
| **Panel regression (Table 4)** | inflation (above) + macro regressors (see *Panel-regression inputs*) | `Panel Regressions Python/Data/Panelh{1,3,6,12}.csv` | upstream assembly scripts (not shipped); the RW−RF loss differential comes from `Replication results/Main case/{RW,RF}/` | `Panel Regressions Python/run_pcse.py` → Table 4 |

Rebuilding `Processed data/` from the raw inflation data requires **MATLAB** (`Data work/`);
regenerating the forecast-error files is the months-long Path B (`Forecasting/`). Per the IJF
guidance, these intermediaries may be used as-is when regenerating them from raw is impractical.

---

## 4. Computing environment

- **Python 3.9** (any 3.9.x; tested on 3.9.13). Required for both Python stages; pinned
  package versions are in `requirements.txt` (forecasting + tables) and
  `requirements-figures.txt` (figures). Python 3.9 is required because
  TensorFlow 2.7.0 (used for the autoencoder factors and the neural net) has no
  wheels for Python ≥ 3.10.
- **MATLAB** — we used R2021a (any recent version should work); only for the
  data-construction step in `Data work/`. Not needed if you use the provided
  `Processed data/` panels.
- **Table 4 (panel regressions)** — reproduced by `Panel Regressions Python/run_pcse.py`
  (Python; needs only `numpy` + `pandas`, listed in `Panel Regressions Python/requirements.txt`).
- **Operating system:** the Python code is OS-independent. The paper's forecasts
  were produced on the grendel Linux HPC cluster (see §7). The package is split
  into a forecasting env and a figures env (the figures env adds the plotting/geo
  libraries `matplotlib`, `shap`, `geopandas` and `mapclassify` on top of the core stack).
- **Key pinned versions** (full list in `requirements.txt`):
  numpy 1.22.4, scipy 1.13.1, scikit-learn 1.0.1, pandas 1.3.5, joblib 1.0.1,
  statsmodels 0.14.1, xgboost 2.1.4, optuna 3.6.1, tensorflow 2.7.0, keras 2.7.0.

---

## 5. Installation

Use [uv](https://github.com/astral-sh/uv) (recommended) or `pip`/`venv`.

**Forecasting + tables env** (`requirements.txt`):
```bash
uv venv --python 3.9 .venv
uv pip install -r requirements.txt          # add --link-mode=copy if on OneDrive/Dropbox
# or:  py -3.9 -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

**Figures env** (`requirements-figures.txt`) — kept separate because it layers
the figure-only libraries (`matplotlib`, `shap`, `geopandas`, `mapclassify`) on
top of the core stack:
```bash
uv venv --python 3.9 .venv-fig
uv pip install --python .venv-fig/Scripts/python.exe -r requirements-figures.txt   # Windows
# macOS/Linux:  uv pip install --python .venv-fig/bin/python -r requirements-figures.txt
```
> The figure scripts only read pre-computed result files, so the figures env does
> not need to match the forecasting pins exactly. Installing `requirements-figures.txt`
> covers all of them; beyond the shared core stack and `matplotlib`, only
> `dependence_plots.py` additionally needs `shap`, and only `rmse_maps.py` needs
> `geopandas` + `mapclassify`.

**Data-construction env:** MATLAB (any version) for `Data work/*.m` (only if
rebuilding `Processed data/` from the raw data).

---

## 6. How to reproduce each table and figure

### Path A — tables & figures from the provided error files (fast, exact)

Run from the repository root with the corresponding env active.

**Tables** (`requirements.txt` env), each writes to `Tables/Results/`:

| Script | Produces |
|---|---|
| `Tables/descriptive_statistics.py`        | Tables 8, 9, 10 |
| `Tables/forecast_evaluation.py`           | Tables 2, 3, 12, 13 |
| `Tables/encompassing_tests.py`            | Table 5 |
| `Tables/rfols_evaluation.py`              | Table 6 |
| `Tables/alternative_factors_evaluation.py`| Table 11 |
| `Tables/case_studies.py`                  | Table 7 |

**Figures** (`requirements-figures.txt` env), each writes to `Figures/Results/`:

| Script | Produces |
|---|---|
| `Figures/rmse_boxplots.py`         | Figures 1, 16 |
| `Figures/cssed_plots.py`           | Figures 2, 17–31 |
| `Figures/rmse_maps.py`             | Figures 3, 9, 10, 11 (RMSE) and 12–15 (RMSE ratios) |
| `Figures/stacked_importance.py`    | Figures 4, 32 |
| `Figures/case_study_importance.py` | Figure 5 |
| `Figures/dependence_plots.py`      | Figure 33 |

> **Table 4** (panel-regression results) is produced by `Panel Regressions Python/run_pcse.py`
> (run `python run_pcse.py` from that folder; it reads `Data/Panelh{1,3,6,12}.csv` and prints the
> coefficients, t-statistics, R² and N for each horizon). **Table 1** and **Figures 6–8** are prepared manually
> (descriptive/illustrative items in the manuscript text) and are not generated by
> code.

### Path B — regenerate the forecast-error files (heavy)

Run the scripts in `Forecasting/Main case/`, `Forecasting/Extended/` and
`Forecasting/Shorter window/` (forecasting + tables env). Each writes its model's
forecast/error workbooks into the matching `Replication results/<sample>/<model>`
folder; then run Path A. `Forecasting/Main case/Get_AutoFactors.py` regenerates
the autoencoder factors used by `Forecast_RF_AE.py`. See §7 for runtime.

---

## 7. Hardware and expected runtime

- **Tables (Path A):** five of the six scripts run in **under ~20 s each**
  (`descriptive_statistics` ~1 s; `encompassing_tests`, `rfols_evaluation`,
  `alternative_factors_evaluation`, `case_studies` 9–21 s).
  `forecast_evaluation.py` is the slowest at **~10 minutes** (it bootstraps the
  multi-horizon SPA test for every country × model). Full table suite ≈ 11 minutes.
- **Figures (Path A):** a few minutes each; the SHAP figures
  (`stacked_importance.py`, `case_study_importance.py`, `dependence_plots.py`)
  read the large `RF_shapvalues_train_only/` folder and take longer.
- **Forecasts (Path B):** very heavy. Even with the ~80 cores available (so that
  several models/samples ran in parallel), regenerating the full set across all
  three samples is estimated at **one to two months** of wall-clock time. Random
  Forest, XGBoost (Optuna tuning) and the neural-net ensemble dominate the cost.
  **This is why the per-country forecast-error files are shipped in
  `Replication results/`:** the reproducibility check runs via Path A (tables &
  figures from the provided errors) in minutes, without re-running the forecasts.

### Servers and hardware

- **Forecasts (Path B)** were produced on the **grendel** HPC cluster at the
  Centre for Scientific Computing Aarhus (CSCAA), with access to **~80 cores
  (≈ four q20 nodes)**. Each individual model run used **one full q20 node
  (20 cores)**, and several runs were executed in parallel. Node hardware
  specifications: <https://www.cscaa.dk/grendel/hardware/>. The code sets
  `n_jobs = 30`, which on a 20-core node simply oversubscribes harmlessly —
  `n_jobs` affects only speed, not the results (forecasts are deterministic given
  the fixed seeds). Reviewers on other machines can set `n_jobs` to their
  available core count (search `Functions.py` and the `Forecast_*.py` scripts).
- **Tables and figures (Path A)** require no special hardware and run in minutes
  on a standard machine.

---

## 8. Special setup

- **Parallelism:** the RF and XGBoost models use `n_jobs = 30` (scikit-learn /
  XGBoost), and the neural-net ensemble fits its 51 members in parallel via
  `joblib.Parallel(n_jobs=18)` in `Forecast_NN.py`. Adjust these to the number of
  cores on the reproduction machine (search for `n_jobs` in `Functions.py` and the
  `Forecast_*.py` scripts). These settings affect only speed, not the results.
- **No GPU required.** TensorFlow is forced onto the CPU
  (`CUDA_VISIBLE_DEVICES=-1` is set at the top of `Functions.py`).

---

## 9. Additional notes

- Table 1 and Figures 6–8 are prepared manually (not script-generated); Table 4 is produced by `Panel Regressions Python/run_pcse.py` — see §6.
- `RF_shapvalues_train_only/` (~16 GB) and `RF_datasets_train_only/` (~0.5 GB)
  are included in the repo and are needed only by the SHAP figures
  (4, 5, 32, 33); they can also be regenerated by `Forecast_RF.py`. See §3.
