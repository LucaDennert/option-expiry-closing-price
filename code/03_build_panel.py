"""Step 3 - Build the stock-day panel (CRSP) and merge the expiration variables.

Key variables
  auc_proxy  log(close / midpoint of CRSP closing bid and ask)   -> closing-price move (proxy)
  r_on       log(next open / close)                               -> overnight return
  adv20      20-day average dollar volume, lagged one day
  gam_int    expiring dollar gamma / adv20                        -> gamma intensity
Output: ~/essay_data/proc/panel_YYYY.parquet
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = os.path.expanduser("~/essay_data/raw/")
P = os.path.expanduser("~/essay_data/proc/")
NUM = ["dlyprc", "dlyopen", "dlyhigh", "dlylow", "dlybid", "dlyask", "dlyvol", "dlyret", "dlycap"]


def crsp_year(y):
    files = [R + f"crsp_dsf_{yy}.parquet" for yy in (y - 1, y, y + 1)]
    d = pd.concat([pd.read_parquet(f) for f in files if os.path.exists(f)])
    d["date"] = pd.to_datetime(d.date)
    for c in NUM:
        d[c] = d[c].astype("float64")
    d = d.sort_values(["permno", "date"])
    d["prc"] = d.dlyprc.abs()
    ok = (d.dlybid > 0) & (d.dlyask > d.dlybid)
    d["mid_q"] = np.where(ok, (d.dlybid + d.dlyask) / 2, np.nan)
    d["hs_rel"] = np.where(ok, (d.dlyask - d.dlybid) / (d.dlybid + d.dlyask), np.nan)
    d["auc_proxy"] = np.log(d.prc / d.mid_q)
    gp = d.groupby("permno")
    d["next_open"] = gp.dlyopen.shift(-1)
    d["r_on"] = np.log(d.next_open / d.prc)
    d["r_cc_prev"] = np.log(d.prc / gp.prc.shift(1))
    d["dvol"] = d.prc * d.dlyvol
    d["adv20"] = gp.dvol.transform(lambda x: x.shift(1).rolling(20, min_periods=10).mean())
    d["rv_hl"] = np.log(d.dlyhigh / d.dlylow)
    d["vol20"] = gp.dlyret.transform(lambda x: x.shift(1).rolling(20, min_periods=10).std())
    return d[d.date.dt.year == y]


def link_table():
    link = pd.read_parquet(R + "link_om_crsp.parquet")
    link = link[link.score == 1]
    for c in ["sdate", "edate"]:
        link[c] = pd.to_datetime(link[c])
    link["secid"] = link.secid.astype("float64")
    return link[["secid", "permno", "sdate", "edate"]]


def panel_year(y):
    d = crsp_year(y)
    link = link_table()

    e = pd.read_parquet(P + f"expiry_level_{y}.parquet").rename(columns={"exdate": "date"})
    e = e.merge(link, on="secid", how="inner")
    e = e[(e.date >= e.sdate) & (e.date <= e.edate)].drop(columns=["sdate", "edate"])
    e = e.sort_values("oi_tot", ascending=False).drop_duplicates(["permno", "date"])

    sd = pd.read_parquet(R + f"om_secday_{y}.parquet")
    sd["date"] = pd.to_datetime(sd.date)
    sd["secid"] = sd.secid.astype("float64")
    sd = sd.merge(link, on="secid")
    sd = sd[(sd.date >= sd.sdate) & (sd.date <= sd.edate)]
    sd = sd.groupby(["permno", "date"])[["oi_all", "goi_all", "vol_all"]].sum().reset_index()

    d = d.merge(sd, on=["permno", "date"], how="left").merge(e, on=["permno", "date"], how="left")
    d["optionable"] = d.oi_all.fillna(0) > 0
    d["is_exp"] = d.n_k.notna()

    sp = pd.read_parquet(R + "crsp_sp500list.parquet")
    for c in ["mbrstartdt", "mbrenddt"]:
        sp[c] = pd.to_datetime(sp[c])
    m = d[["permno", "date"]].merge(sp[["permno", "mbrstartdt", "mbrenddt"]], on="permno")
    m = m[(m.date >= m.mbrstartdt) & (m.date <= m.mbrenddt)][["permno", "date"]].drop_duplicates()
    m["sp500"] = True
    d = d.merge(m, on=["permno", "date"], how="left")
    d["sp500"] = d.sp500.fillna(False)

    d["gam_int"] = (d.dg_c.fillna(0) + d.dg_p.fillna(0)) / d.adv20
    d["dist_maxoi"] = (d.k_maxoi - d.mid_q) / d.mid_q
    d.to_parquet(P + f"panel_{y}.parquet", index=False)
    return len(d), int(d.is_exp.sum())


if __name__ == "__main__":
    for y in range(2016, 2026):
        print(y, panel_year(y))
