"""
Forecast encompassing tests (Table 5).

For each country and horizon, runs the Harvey, Leybourne, and Newbold
(1998) forecast encompassing test on the RF forecast based on the
high-dimensional inflation panel (I, the "OnlyI" error files) versus
the RF forecast based on the global and regional factors (F, the
"OnlyF" error files), using the stored forecast errors.

The table reports, per horizon:

    rows (1)-(2)  average combination weights lambda_I and lambda_F
                  across the 91 countries
    rows (3)-(6)  share of countries (in percent) in each significance
                  scenario at alpha = 0.05, where P_I and P_F are the
                  country-specific p-values of the nulls lambda_I = 0
                  and lambda_F = 0
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERRORS_DIR = ROOT / "Replication results" / "Main case" / "Encompassing"
RESULTS_DIR = ROOT / "Tables" / "Results"

sys.path.insert(0, str(ROOT))
from Functions import _EncompassingTest  # noqa: E402

HORIZONS = [1, 3, 6, 12]
ALPHA = 0.05

table = {}
for h in HORIZONS:
    e_I = pd.read_excel(ERRORS_DIR / f"e_RF_OnlyI_h{h}.xlsx",
                        index_col="Date")
    e_F = pd.read_excel(ERRORS_DIR / f"e_RF_OnlyF_h{h}.xlsx",
                        index_col="Date")
    assert list(e_I.columns) == list(e_F.columns), "country columns differ"

    res = pd.DataFrame(
        [_EncompassingTest(e_I[c], e_F[c], h) for c in e_I.columns],
        index=e_I.columns, columns=["lambda_I", "p_I", "lambda_F", "p_F"],
    )
    sig_I = res["p_I"] <= ALPHA
    sig_F = res["p_F"] <= ALPHA
    table[f"h = {h}"] = {
        "Average lambda_I": res["lambda_I"].mean(),
        "Average lambda_F": res["lambda_F"].mean(),
        "P_I <= a, P_F > a, share of countries": (sig_I & ~sig_F).mean() * 100,
        "P_I > a, P_F <= a, share of countries": (~sig_I & sig_F).mean() * 100,
        "P_I <= a, P_F <= a, share of countries": (sig_I & sig_F).mean() * 100,
        "P_I > a, P_F > a, share of countries": (~sig_I & ~sig_F).mean() * 100,
    }

table = pd.DataFrame(table)
table.iloc[:2] = table.iloc[:2].round(3)
table.iloc[2:] = table.iloc[2:].round(0)
table.to_excel(RESULTS_DIR / "Table5_EncompassingTests.xlsx")
print(table)
