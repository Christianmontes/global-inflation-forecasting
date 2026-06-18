# Reproducibility package — *Global Inflation Forecasting*

International Journal of Forecasting, manuscript IJF-D-25-00472R1.

- **Package assembled:** 2026-06-18
- **Authors of the paper:** Marcelo C. Medeiros, Erik Christian Montes Schutte, and Tobias Skipper Soussi
- **Package author / contact for the reproducibility check:** Tobias Skipper Soussi — tss@econ.au.dk

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
  **exactly** (verified bit-for-bit; see §7).
- **Path B — regenerate the forecasts (very heavy).** Re-run the scripts in
  `Forecasting/` to recompute the forecast-error files, then run Path A. Most
  models reproduce to machine precision; a few do not reproduce *bit-for-bit* for
  the reasons documented in §7, but the differences are tiny and do not change
  any conclusion in the paper.

---

## 2. Repository structure

```
.
├── README.md                     this file
├── requirements.txt              Python env for Forecasting/ and Tables/ (pinned)
├── requirements-figures.txt      Python env for Figures/ (separate; see §5)
├── install_packages.R            R packages for the Table 4 regressions (see §6)
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
│   │                             (last two are large; not in Git — see §3)
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
└── Global_Inflation_Forecasting.pdf   the manuscript
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
- **Map shapefile** (`Figures/ne_10m_admin_0_countries/`): Natural Earth 1:10m
  Admin-0 countries, used only by `rmse_maps.py`. Natural Earth data is in the
  **public domain**; the files are bundled for convenience.
- **Intermediary forecast outputs** (`Replication results/`): per-country
  forecast and forecast-error workbooks, one per model and horizon — the output
  of `Forecasting/` (Path B) and the input to `Tables/`/`Figures/` (Path A).
  Also here: `AE_factors/` (autoencoder factors, see §7),
  `RF_shapvalues_train_only/` and `RF_datasets_train_only/` (per-window SHAP
  values and training frames used only by the SHAP figures).
  - **Large intermediary files — not included in the Git repository.**
    `RF_shapvalues_train_only/` (~17 GB) and `RF_datasets_train_only/` (~0.5 GB)
    are per-window SHAP values and training frames used **only** by the four SHAP
    figures (4, 5, 32, 33). They are too large for GitHub and are **regenerated by
    `Forecasting/Main case/Forecast_RF.py`**, which writes them as it runs. Per the
    IJF guidance, such large intermediary files may be omitted from the kit and the
    rest of the verification still carried out — only those four figures need them.
    [TODO: decide whether to host these two folders on Zenodo/OSF (for a DOI) and
    link them here, or leave them as regenerated-only.]

---

## 4. Computing environment

- **Python 3.9** (exactly 3.9.25 used). Required for both Python stages; pinned
  package versions are in `requirements.txt` (forecasting + tables) and
  `requirements-figures.txt` (figures). Python 3.9 is required because
  TensorFlow 2.7.0 (used for the autoencoder factors and the neural net) has no
  wheels for Python ≥ 3.10.
- **MATLAB** — we used R2021a (any recent version should work); only for the
  data-construction step in `Data work/`. Not needed if you use the provided
  `Processed data/` panels.
- **R** — used for the panel-regression results in **Table 4** (R packages in
  `install_packages.R`). [TODO: R version, the R script that produces Table 4, and
  how to run it.]
- **Operating system:** the Python code is OS-independent. The paper's forecasts
  were produced on the grendel Linux HPC cluster (see §8). The package is split
  into a forecasting env and a figures env (the figures need `geopandas`, which
  requires a newer numpy than the pinned forecasting stack).
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

**Figures env** (`requirements-figures.txt`) — kept separate because the maps
figure needs `geopandas`, which requires a newer numpy than the pinned
forecasting stack:
```bash
uv venv --python 3.9 .venv-fig
uv pip install --python .venv-fig/Scripts/python.exe -r requirements-figures.txt   # Windows
# macOS/Linux:  uv pip install --python .venv-fig/bin/python -r requirements-figures.txt
```
> The figure scripts only read pre-computed result files, so the figures env does
> not need to match the forecasting pins exactly. `cssed_plots.py`,
> `rmse_boxplots.py`, `case_study_importance.py`, `stacked_importance.py` and
> `dependence_plots.py` need only `matplotlib` (+ `shap` for the SHAP figures);
> `rmse_maps.py` additionally needs `geopandas` + `mapclassify`.

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

> **Table 4** (panel-regression results) is produced by separate **R** code, not
> by the Python scripts above. [TODO: add the R script(s) for Table 4 and how to
> run them.] **Table 1** and **Figures 6–8** are prepared manually
> (descriptive/illustrative items in the manuscript text) and are not generated by
> code.

### Path B — regenerate the forecast-error files (heavy)

Run the scripts in `Forecasting/Main case/`, `Forecasting/Extended/` and
`Forecasting/Shorter window/` (forecasting + tables env). Each writes its model's
forecast/error workbooks into the matching `Replication results/<sample>/<model>`
folder; then run Path A. `Forecasting/Main case/Get_AutoFactors.py` regenerates
the autoencoder factors used by `Forecast_RF_AE.py`. See §7 for what does and
does not reproduce bit-for-bit, and §8 for runtime.

---

## 7. Reproducibility notes — what matches and what does not

**Path A reproduces the provided tables and figures exactly** — running the
`Tables/`/`Figures/` scripts on the provided `Replication results/` regenerates
Tables 2–13 bit-for-bit and redraws every figure from the same stored files.
**Path B (regenerating the forecasts)** reproduces the closed-form/linear models —
RW, SRW, dummy-augmented AR, Ciccarelli–Mojon, and (with scikit-learn 1.0.x) the
Elastic Net — to machine precision. The machine-learning results differ only by
small, well-understood, **non-material** amounts that change no conclusion: a
`max_features` cross-validation tie-break in Random Forest and its PCA/DFM/OLS
variants (~1e-3 per affected window; the reported ratios round the same), and the
originally **unseeded** XGBoost, neural-net ensemble, regenerated autoencoder
factors, and multi-horizon SPA-test bootstrap (whose p-values sit in the last
columns of Tables 2, 3, 12 and 13). The code here now fixes all of these seeds, so
the results are reproducible run-to-run going forward.

---

## 8. Hardware and expected runtime

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

## 9. Special setup

- **Parallelism:** the RF and XGBoost models use `n_jobs = 30` (scikit-learn /
  XGBoost), and the neural-net ensemble fits its 51 members in parallel via
  `joblib.Parallel(n_jobs=18)` in `Forecast_NN.py`. Adjust these to the number of
  cores on the reproduction machine (search for `n_jobs` in `Functions.py` and the
  `Forecast_*.py` scripts). These settings affect only speed, not the results.
- **No GPU required.** TensorFlow is forced onto the CPU
  (`CUDA_VISIBLE_DEVICES=-1` is set at the top of `Functions.py`).

---

## 10. Known issues / housekeeping

- `install_packages.R` lists the R packages for the **Table 4** panel regressions. [TODO: add the R script(s) that produce Table 4 and run instructions.]
- Table 1 and Figures 6–8 are prepared manually (not script-generated); Table 4 is produced by R — see §6.
- `RF_shapvalues_train_only/` (~17 GB) and `RF_datasets_train_only/` (~0.5 GB)
  are not in the Git repo (too large); they are regenerated by `Forecast_RF.py`
  and are needed only by the SHAP figures (4, 5, 32, 33). See §3. [TODO: decide
  whether to host them on Zenodo/OSF and link.]
```
