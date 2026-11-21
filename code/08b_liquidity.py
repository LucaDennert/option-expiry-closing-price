"""Step 8b - |closing move| among optionable stocks with date x liquidity-decile fixed effects.

Sample construction identical to 08_robustness.py (copied below).

Step 8 - Revisions after referee-style feedback.

A. Pinning, redone so that it does not rest on gamma (which is mechanically high when the price starts near a strike):
   * intensity measured with open interest only: OI at K* (x100 shares) / 20-day average share volume
   * the same measure for treated (expiring OI) and placebo (next expiration's OI) stock-days, so that the slope
     can be estimated in both groups and compared (Exp x OI-rank is the DIFFERENCE in slopes)
   * fixed effects for 10-bp bins of the starting distance |prev close - K*| interacted with volatility terciles
   * date fixed effects (or date x size-decile), stock fixed effects, early-close days excluded
B. Size / liquidity: |closing move| with date x size-decile and date x liquidity-decile fixed effects, optionable
   stocks only, and names without weekly expirations.
C. Corrected magnitudes: model-based percentage effect on |closing move|; total reversal slopes with 95% CIs.
Output: ~/essay_data/results/table6_robustness.json
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
CTRL = ["lmcap", "ladv", "hs_bp", "vol20"]
CTRL_PRE = ["lmcap_l", "ladv", "vol20"]  # predetermined controls only (no same-day spread)
COLS = ["permno", "date", "prc", "mid_q", "hs_rel", "auc_proxy", "r_on", "r_cc_prev", "adv20", "vol20", "dlycap",
        "dlyvol", "optionable", "is_exp", "gam_int", "k_maxoi", "oi_maxk", "ei"]
# NYSE early closes (13:00) in the sample period
EARLY = pd.to_datetime(["2016-11-25", "2017-07-03", "2017-11-24", "2018-07-03", "2018-11-23", "2018-12-24",
                        "2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27", "2020-12-24", "2021-11-26",
                        "2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
                        "2025-07-03"])


def wins(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def run(res, name, dep, exog, df, fe="date", extra_fe=None):
    """Stock FE + date FE, or stock FE + one combined categorical FE; two-way clustered SEs (stock, day)."""
    need = [dep] + exog + ([fe] if fe != "date" else [])
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    exog = [c for c in exog if d[c].abs().sum() > 0]  # drop dummies that are empty in a sub-sample
    if extra_fe is not None:  # absorb an additional categorical by combining it with the time dimension
        d["_fe"] = pd.factorize(d["date"].astype(str) + "_" + d[extra_fe].astype(str))[0]
        fe = "_fe"
    d = d.set_index(["permno", "date"])
    clusters = pd.DataFrame({"stock": d.index.get_level_values(0).astype("int64"),
                             "day": pd.factorize(d.index.get_level_values(1))[0]}, index=d.index)
    if fe == "date":
        mod = PanelOLS(d[dep], d[exog], entity_effects=True, time_effects=True)
    else:
        mod = PanelOLS(d[dep], d[exog], entity_effects=True, other_effects=d[[fe]])
    r = mod.fit(cov_type="clustered", clusters=clusters)
    res[name] = {"nobs": int(r.nobs), "params": r.params.to_dict(), "t": r.tstats.to_dict(),
                 "cov": r.cov.to_dict()}
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "cov"} if isinstance(v, dict) else v for k, v in res.items()},
              open(OUT + "table6b_liquidity.json", "w"), indent=1, default=float)
    print(name, int(r.nobs), {k: (round(r.params[k], 4), round(r.tstats[k], 2)) for k in exog[:3]}, flush=True)
    return r


# ------------------------------------------------------------------ sample
D = pd.concat([pd.read_parquet(P + f"panel_{y}.parquet", columns=COLS) for y in range(2016, 2026)])
D = D[D.date <= "2025-08-29"]
D["prc_prev"] = D.prc / np.exp(D.r_cc_prev)
D = D[(D.prc_prev > 5) & D.mid_q.notna() & D.adv20.notna()]
D = D.sort_values(["permno", "date"])
D["dow"] = D.date.dt.dayofweek
D["exp_day"] = D.date.isin(set(D.loc[D.is_exp, "date"].unique()))
D["auc_bp"] = wins(D.auc_proxy * 1e4)
D["on_bp"] = wins(D.r_on * 1e4)
D["abs_auc"] = D.auc_bp.abs()
D["lmcap"] = np.log(D.dlycap)
D["lmcap_l"] = D.groupby("permno").lmcap.shift(1)
D["ladv"] = np.log(D.adv20)
D["hs_bp"] = D.hs_rel * 1e4
D["shvol20"] = D.groupby("permno").dlyvol.transform(lambda x: x.shift(1).rolling(20, min_periods=10).mean())

F = D[((D.dow == 4) | D.exp_day) & ~D.date.isin(EARLY)].copy()
F["gi"] = F.gam_int.where(F.is_exp, 0).fillna(0)
F["gir"] = 0.0
F.loc[F.is_exp, "gir"] = F[F.is_exp].groupby("date").gi.rank(pct=True)
F["expx"] = F.is_exp.astype(float)
F["exp_gir"] = F.expx * F.gir
F["size_dec"] = F.groupby("date").dlycap.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
F["liq_dec"] = F.groupby("date").adv20.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))

# placebo strike and its open interest: next expiration's max-OI strike on Fridays without expiring options
link = pd.read_parquet(R + "link_om_crsp.parquet")
link = link[link.score == 1]  # WRDS OptionMetrics-CRSP link: score 1 (best) to 6 (worst)
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
F = F.merge(G[["permno", "date", "k_next", "oi_next"]], on=["permno", "date"], how="left")
plc = F.optionable & ~F.is_exp
F["kstar"] = np.where(F.is_exp, F.k_maxoi, np.where(plc, F.k_next, np.nan))
F["oi_k"] = np.where(F.is_exp, F.oi_maxk, np.where(plc, F.oi_next, np.nan))
F["prev_close"] = F.prc / np.exp(F.r_cc_prev)
F["adist_prev"] = (F.prev_close - F.kstar).abs() / F.kstar * 1e4
F["adist_close"] = (F.prc - F.kstar).abs() / F.kstar * 1e4
F["pin10"] = (F.adist_close <= 10).astype(float)
F["pin25"] = (F.adist_close <= 25).astype(float)
F["toward_bp"] = F.adist_prev - F.adist_close  # positive = closer to K* at the close than at the previous close

res = {}
X = ["expx", "exp_gir"] + CTRL
run(res, "B_optionable_date_x_liquidity", "abs_auc", X, F[F.optionable], extra_fe="liq_dec")
run(res, "B_optionable_date_x_liquidity_nocontrols", "abs_auc", ["expx", "exp_gir"], F[F.optionable], extra_fe="liq_dec")
Fo = F[F.optionable].copy()
Fo["liq_dec_o"] = Fo.groupby("date").adv20.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
run(res, "B_optionable_date_x_liqdec_within_optionable", "abs_auc", X, Fo, extra_fe="liq_dec_o")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "cov"} for k, v in res.items()}, open(OUT + "table6b_liquidity.json", "w"), indent=1, default=float)
print("done", flush=True)
