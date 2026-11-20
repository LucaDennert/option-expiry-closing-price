"""Step 5 - Estimation (Tables 1-5 and Figure 1).

Sample: CRSP common stocks, Jan 2016 - 29 Aug 2025, previous close > $5.
All regressions: stock and date fixed effects, SEs clustered by stock and date (linearmodels.PanelOLS).
Output: ~/essay_data/results/*.json, *.csv
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")
P = os.path.expanduser("~/essay_data/proc/")
R = os.path.expanduser("~/essay_data/raw/")
OUT = os.path.expanduser("~/essay_data/results/")
os.makedirs(OUT, exist_ok=True)
CTRL = ["lmcap", "ladv", "hs_bp", "vol20"]
COLS = ["permno", "date", "prc", "dlyopen", "mid_q", "hs_rel", "auc_proxy", "r_on", "r_cc_prev", "adv20",
        "vol20", "rv_hl", "dlycap", "optionable", "is_exp", "sp500", "gam_int", "dist_maxoi", "k_maxoi",
        "S", "oi_tot", "ei", "n_k", "dg_c", "dg_p", "secid"]


def wins(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def run(res, name, dep, exog, df):
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[dep] + exog).set_index(["permno", "date"])
    r = PanelOLS(d[dep], d[exog], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True)
    res[name] = {"nobs": int(r.nobs), "params": r.params.to_dict(), "t": r.tstats.to_dict()}
    print(name, int(r.nobs), {k: (round(r.params[k], 3), round(r.tstats[k], 2)) for k in exog})
    return r


# ---------------------------------------------------------------- sample
D = pd.concat([pd.read_parquet(P + f"panel_{y}.parquet", columns=COLS) for y in range(2016, 2026)])
D = D[D.date <= "2025-08-29"]
D["prc_prev"] = D.prc / np.exp(D.r_cc_prev)
D = D[(D.prc_prev > 5) & D.mid_q.notna() & D.adv20.notna()]
D["dow"] = D.date.dt.dayofweek
D["exp_day"] = D.date.isin(set(D.loc[D.is_exp, "date"].unique()))
D["auc_bp"] = wins(D.auc_proxy * 1e4)
D["on_bp"] = wins(D.r_on * 1e4)
D["day_bp"] = wins(D.r_cc_prev * 1e4)
D["abs_auc"] = D.auc_bp.abs()
D["abs_on"] = D.on_bp.abs()
D["lmcap"] = np.log(D.dlycap)
D["ladv"] = np.log(D.adv20)
D["hs_bp"] = D.hs_rel * 1e4

# Fridays (+ Thursdays on which options expire)
F = D[(D.dow == 4) | D.exp_day].copy()
F["gi"] = F.gam_int.where(F.is_exp, 0).fillna(0)
F["gir"] = 0.0
F.loc[F.is_exp, "gir"] = F[F.is_exp].groupby("date").gi.rank(pct=True)  # rank among same-day expirations
F["expx"] = F.is_exp.astype(float)
F["exp_gir"] = F.expx * F.gir

# ---------------------------------------------------------------- Table 1
F["grp"] = np.where(F.is_exp, "exp", np.where(F.optionable, "opt_noexp", "nonopt"))
t1 = F.groupby("grp").agg(obs=("permno", "size"), stocks=("permno", "nunique"), abs_auc=("abs_auc", "mean"),
                          abs_on=("abs_on", "mean"), hs_bp=("hs_bp", "median"),
                          mcap_bn=("dlycap", lambda s: s.median() / 1e6))
t1.to_csv(OUT + "table1_descriptives.csv")

# ---------------------------------------------------------------- Table 3: size of closing moves
res = {}
run(res, "abs_auc", "abs_auc", ["expx", "exp_gir"] + CTRL, F)
run(res, "abs_on", "abs_on", ["expx", "exp_gir"] + CTRL, F)
json.dump(res, open(OUT + "table3_magnitude.json", "w"), indent=1)

# ---------------------------------------------------------------- placebo strike: next expiration's max-OI strike
link = pd.read_parquet(R + "link_om_crsp.parquet")
link = link[link.score == 1]
for c in ["sdate", "edate"]:
    link[c] = pd.to_datetime(link[c])
link["secid"] = link.secid.astype("float64")
G = []
for y in range(2016, 2026):
    g = pd.read_parquet(R + f"om_friday_nextgrid_{y}.parquet")
    g["secid"] = g.secid.astype("float64")
    g["date"] = pd.to_datetime(g.date)
    for c in ["k", "oi_c", "oi_p"]:
        g[c] = g[c].astype("float64")
    g["oi_t"] = g.oi_c + g.oi_p
    i = g.groupby(["secid", "date"]).oi_t.idxmax()
    G.append(g.loc[i.values, ["secid", "date", "k", "oi_t"]].rename(columns={"k": "k_next", "oi_t": "oi_next"}))
G = pd.concat(G).merge(link[["secid", "permno", "sdate", "edate"]], on="secid")
G = G[(G.date >= G.sdate) & (G.date <= G.edate)].sort_values("oi_next", ascending=False)
G = G.drop_duplicates(["permno", "date"])
F = F.merge(G[["permno", "date", "k_next"]], on=["permno", "date"], how="left")
F["kstar"] = np.where(F.is_exp, F.k_maxoi, np.where(F.optionable & ~F.is_exp, F.k_next, np.nan))
F["prev_close"] = F.prc / np.exp(F.r_cc_prev)
F["adist_prev"] = (F.prev_close - F.kstar).abs() / F.kstar * 1e4
F["adist_close"] = (F.prc - F.kstar).abs() / F.kstar * 1e4
F["adist_mid"] = (F.mid_q - F.kstar).abs() / F.kstar * 1e4
F["pin10"] = (F.adist_close <= 10).astype(float)
F["pin25"] = (F.adist_close <= 25).astype(float)
F["dred_day"] = F.adist_close - F.adist_prev
F["dred_auc"] = F.adist_close - F.adist_mid

# ---------------------------------------------------------------- Table 2: pinning
res = {}
for band in [100, 200]:
    S = F[F.kstar.notna() & (F.adist_prev <= band)]
    res[f"placebo_rate_{band}"] = S[~S.is_exp][["pin10", "pin25"]].mean().to_dict()
    for dep in ["pin10", "pin25", "dred_day", "dred_auc"]:
        run(res, f"band{band}_{dep}", dep, ["expx", "exp_gir", "hs_bp", "vol20"], S)
S = F[F.kstar.notna() & (F.adist_prev <= 200)]
for lab, cond in [("std", (~S.is_exp) | (S.ei == "std")), ("weekly", (~S.is_exp) | (S.ei == "w")),
                  ("2016_19", S.date < "2020-01-01"), ("2020_25", S.date >= "2020-01-01")]:
    run(res, "robust_pin25_" + lab, "pin25", ["expx", "exp_gir", "hs_bp", "vol20"], S[cond])
json.dump(res, open(OUT + "table2_pinning.json", "w"), indent=1)

# Figure 1 data: pin25 by gamma-intensity quintile (expirations) and placebo rate
x = S[S.is_exp].copy()
x["q"] = np.ceil(x.gir * 5).clip(1, 5).astype(int)
fig = x.groupby("q").agg(pin25=("pin25", "mean"), n=("pin25", "size"), gi_med=("gi", "median"))
fig.loc["placebo"] = [S[~S.is_exp].pin25.mean(), (~S.is_exp).sum(), np.nan]
fig.to_csv(OUT + "figure1_pinning_by_quintile.csv")

# ---------------------------------------------------------------- Table 4: overnight reversals
res = {}
S = F[F.auc_bp.notna() & F.on_bp.notna()].copy()
S["auc"] = S.auc_bp
S["auc_x_exp"] = S.auc * S.expx
S["auc_x_exp_gir"] = S.auc * S.exp_gir
run(res, "rev_auc", "on_bp", ["auc", "auc_x_exp", "auc_x_exp_gir", "expx", "exp_gir"], S)
S["day"] = S.day_bp
S["day_x_exp"] = S.day * S.expx
S["day_x_exp_gir"] = S.day * S.exp_gir
run(res, "rev_day", "on_bp", ["day", "day_x_exp", "day_x_exp_gir", "expx", "exp_gir"], S)
Q = F[F.kstar.notna() & (F.adist_prev <= 200) & F.on_bp.notna()].copy()
Q["toward"] = -Q.dred_day
Q["on_toward"] = Q.on_bp * np.sign(Q.kstar - Q.prev_close)
Q["tw_x_exp"] = Q.toward * Q.expx
Q["tw_x_exp_gir"] = Q.toward * Q.exp_gir
run(res, "rev_toward", "on_toward", ["toward", "tw_x_exp", "tw_x_exp_gir", "expx", "exp_gir"], Q)
json.dump(res, open(OUT + "table4_reversal.json", "w"), indent=1)


# ---------------------------------------------------------------- Table 5: SPX Tuesday/Thursday expirations
def spx_sample(start, end, post_date, intro_tue, intro_thu):
    E = D[(D.date >= start) & (D.date <= end) & D.dow.isin([0, 1, 2, 3]) & D.optionable].copy()
    E = E[E.sp500 | (E.dlycap > 2e6)]  # members vs non-members above $2bn (dlycap in $000)
    E["sp"] = E.sp500.astype(float)
    E["tt"] = E.dow.isin([1, 3]).astype(float)
    E["post"] = (E.date >= post_date).astype(float)
    E["ttpost"] = (((E.dow == 1) & (E.date >= intro_tue)) | ((E.dow == 3) & (E.date >= intro_thu))).astype(float)
    E["sp_tt"] = E.sp * E.tt
    E["sp_post"] = E.sp * E.post
    E["sp_ttpost"] = E.sp * E.ttpost
    return E


res = {}
# listing from 18 Apr 2022; first Tuesday / Thursday SPXW expirations on 26 Apr and 19 May 2022
E = spx_sample("2021-01-01", "2023-06-30", "2022-04-18", "2022-04-26", "2022-05-19")
res["base_abs_auc_sp_tt_pre"] = E[E.sp500 & (E.tt == 1) & (E.post == 0)].abs_auc.mean()
for dep in ["abs_auc", "abs_on"]:
    run(res, "spx_" + dep, dep, ["sp_tt", "sp_post", "sp_ttpost"] + CTRL, E)
E2 = spx_sample("2020-01-01", "2022-03-31", "2021-04-19", "2021-04-19", "2021-04-19")  # placebo: a year earlier
run(res, "spx_placebo_abs_auc", "abs_auc", ["sp_tt", "sp_post", "sp_ttpost"] + CTRL, E2)
json.dump(res, open(OUT + "table5_spx.json", "w"), indent=1, default=float)
print("done")
