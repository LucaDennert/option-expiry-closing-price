"""Step 4 - Data checks reported in Appendix A.2.

* OptionMetrics underlying close on the day before expiration vs CRSP close (within 0.5%)
* weekday of expirations, share of weekly expirations
* missing closing midpoint / next-day open
Output: ~/essay_data/verification_summary.csv
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
P = os.path.expanduser("~/essay_data/proc/")
COLS = ["permno", "date", "prc", "mid_q", "r_on", "optionable", "is_exp", "gam_int", "S", "prev_date", "ei"]

rows = []
for y in range(2016, 2026):
    d = pd.read_parquet(P + f"panel_{y}.parquet", columns=COLS)
    x = d[d.prc > 5]
    e = d[d.is_exp]
    c = d[["permno", "date", "prc"]].rename(columns={"date": "prev_date", "prc": "prc_prev"})
    ee = e.merge(c, on=["permno", "prev_date"], how="left")
    rows.append(dict(
        year=y, stock_days=len(d), permnos=d.permno.nunique(), sh_optionable=x.optionable.mean(),
        exp_obs=len(e), exp_dates=e.date.nunique(), sh_exp_friday=(e.date.dt.dayofweek == 4).mean(),
        sh_weekly=(e.ei == "w").mean(), om_crsp_close_agree=(np.abs(ee.S / ee.prc_prev - 1) < 0.005).mean(),
        miss_mid=x.mid_q.isna().mean(), miss_next_open=x.r_on.isna().mean(), gam_int_med=e.gam_int.median(),
    ))
out = pd.DataFrame(rows)
out.to_csv(os.path.expanduser("~/essay_data/verification_summary.csv"), index=False)
print(out.round(3).to_string())
