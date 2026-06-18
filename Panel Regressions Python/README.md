# Panel Regressions (Python) — Table 4

Self-contained Python reproduction of the panel regressions behind **Table 4**
(determinants of forecasting outperformance). Replaces the earlier R/Stata code.

## What it does
Two-way fixed-effects OLS with **Beck-Katz pairwise panel-corrected standard
errors** (PCSE, `nmk` small-sample correction) — the Python equivalent of the
Stata command

```
xtpcse lossdhH openess D.gdp D.gdp_percapita avginf unemp i.year i.id, pairwise nmk
```

For each forecast horizon `h ∈ {1, 3, 6, 12}` it fits the openness-only and the
full (with-controls) specifications and prints coefficients, PCSE t-statistics,
R², and the sample size `N`.

## Files
```
run_pcse.py        the script
requirements.txt   pandas, numpy
Data/              Panelh1.csv, Panelh3.csv, Panelh6.csv, Panelh12.csv
```

## Run
```bash
pip install -r requirements.txt
python run_pcse.py
```

## Notes
- **Validated against R.** Output is identical to the original `run_pcse.R`
  (R 4.6.0) to ~1e-7 on every coefficient, SE, t-stat, N and k.
- **Sample alignment.** The openness-only model is reported on two samples: its
  own (`orig`, reproduces the originally submitted table) and the full-model
  sample (`align`, the sample-consistent version used in the paper). They differ
  because the full model first-differences GDP (dropping each country's first
  year, 2000) and requires the controls to be non-missing — ~108 fewer obs.
- The `NaNs produced` situation for a few fixed-effect dummies is benign and also
  occurs in R; it never affects a reported coefficient.
- Panel: 86 countries, 2000–2019.
