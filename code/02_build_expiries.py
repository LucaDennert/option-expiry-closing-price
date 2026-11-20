"""Step 2 - Build stock-expiration variables from OptionMetrics.

For every underlying and expiration date:
  * open interest by strike at the close before expiration (OptionMetrics reports OI with a one-day lag,
    so the value on the expiration date is the previous close),
  * Black-Scholes gamma with one trading day to expiry, recomputed because OptionMetrics leaves gamma missing
    on the expiration day,
  * expiring dollar gamma for a 1% move, max-OI strike, nearest strike.
Output: ~/essay_data/proc/strikes_exp_YYYY.parquet and expiry_level_YYYY.parquet
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = os.path.expanduser("~/essay_data/raw/")
P = os.path.expanduser("~/essay_data/proc/")
T1 = 1 / 252  # one trading day to expiry


def bs_gamma(S, K, sig, T=T1):
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / sq
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * sq)


def build_year(y):
    s = pd.read_parquet(R + f"om_nearexp_strikes_{y}.parquet")
    for c in ["secid", "k", "oi", "iv", "am"]:
        s[c] = s[c].astype("float64")
    s["date"] = pd.to_datetime(s.date)
    s["exdate"] = pd.to_datetime(s.exdate)
    s["dte"] = (s.exdate - s.date).dt.days

    sec = pd.read_parquet(R + "om_securd.parquet")[["secid", "issue_type"]]
    sec["secid"] = sec.secid.astype("float64")
    s = s.merge(sec, on="secid", how="left")
    s = s[s.issue_type == "0"]  # common stocks only

    key3 = ["secid", "exdate", "k"]
    e0 = s[s.dte == 0].pivot_table(index=key3, columns="cp_flag", values="oi", aggfunc="sum")
    e0 = e0.rename(columns={"C": "oi_c", "P": "oi_p"})
    e1 = s[s.dte == 1]
    iv1 = e1.groupby(key3).iv.mean().rename("iv")
    oi1 = e1.pivot_table(index=key3, columns="cp_flag", values="oi", aggfunc="sum")
    oi1 = oi1.rename(columns={"C": "oi_c1", "P": "oi_p1"})
    g = e0.join(oi1, how="outer").join(iv1, how="left").reset_index()
    for c in ["oi_c", "oi_p"]:
        g[c] = g[c].fillna(g[c + "1"])  # fall back to the E-1 report if the expiry-day row is missing
    g = g.drop(columns=["oi_c1", "oi_p1"]).fillna({"oi_c": 0, "oi_p": 0})

    # underlying close on the trading day before expiration
    px = pd.read_parquet(R + f"om_secprd_{y}.parquet")[["secid", "date", "close"]]
    px["secid"] = px.secid.astype("float64")
    px["date"] = pd.to_datetime(px.date)
    px["close"] = px.close.astype("float64").abs()
    px = px.sort_values(["secid", "date"])
    px["prev_date"] = px.groupby("secid").date.shift(1)
    px["prev_close"] = px.groupby("secid").close.shift(1)
    px = px.rename(columns={"date": "exdate", "prev_close": "S"})
    g = g.merge(px[["secid", "exdate", "prev_date", "S"]], on=["secid", "exdate"], how="left")
    g = g[g.S > 0]

    # implied volatility: the contract's own, else the median IV of strikes within +-10% of spot
    atm = g[(g.k / g.S).between(0.9, 1.1)].groupby(["secid", "exdate"]).iv.median().rename("iv_atm")
    g = g.merge(atm, on=["secid", "exdate"], how="left")
    g["sig"] = g.iv.fillna(g.iv_atm).clip(0.05, 3.0)
    g["gam"] = bs_gamma(g.S, g.k, g.sig)
    g["dg_c"] = g.gam * g.oi_c * 100 * g.S ** 2 * 0.01  # dollar gamma for a 1% move
    g["dg_p"] = g.gam * g.oi_p * 100 * g.S ** 2 * 0.01
    g["oi_t"] = g.oi_c + g.oi_p
    g.to_parquet(P + f"strikes_exp_{y}.parquet", index=False)

    key = ["secid", "exdate"]
    g["adiff"] = (g.k - g.S).abs()
    imax = g.groupby(key).oi_t.idxmax()
    inear = g.groupby(key).adiff.idxmin()
    e = g.groupby(key).agg(
        S=("S", "first"), prev_date=("prev_date", "first"), n_k=("k", "size"), oi_tot=("oi_t", "sum"),
        dg_c=("dg_c", "sum"), dg_p=("dg_p", "sum"), sig_atm=("iv_atm", "first"),
        gam_missing_share=("sig", lambda x: x.isna().mean()),
    )
    e["k_maxoi"] = g.loc[imax.values, "k"].values
    e["oi_maxk"] = g.loc[imax.values, "oi_t"].values
    e["sh_maxoi"] = e.oi_maxk / e.oi_tot.clip(lower=1)
    e["k_near"] = g.loc[inear.values, "k"].values
    e = e.reset_index()
    typ = s.groupby(key).ei.agg(lambda x: x.dropna().iloc[0] if x.notna().any() else "std").rename("ei")
    e = e.merge(typ.reset_index(), on=key, how="left")
    e.to_parquet(P + f"expiry_level_{y}.parquet", index=False)
    return len(g), len(e)


if __name__ == "__main__":
    os.makedirs(P, exist_ok=True)
    for y in range(2016, 2026):
        print(y, build_year(y))
