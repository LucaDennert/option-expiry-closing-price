"""Step 8 - Revisions after referee-style feedback.

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
              open(OUT + "table6_robustness.json", "w"), indent=1, default=float)
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
# ------------------------------------------------------------------ A. pinning
S = F[F.kstar.notna() & (F.adist_prev <= 200) & F.shvol20.gt(0)].copy()
S["oi_int"] = S.oi_k * 100 / S.shvol20                       # OI at K* in shares / normal daily share volume
S["oir"] = S.groupby("date").oi_int.rank(pct=True)          # pooled rank: same scale for treated and placebo
S["exp_oir"] = S.expx * S.oir
S["dbin"] = (S.adist_prev // 10).clip(upper=19).astype(int)  # 10-bp bins of starting distance
S["vterc"] = S.groupby("date").vol20.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
S["dv"] = S.dbin.astype(str) + "_" + S.vterc.astype(str)
res["pin_base_rates"] = {"placebo_pin25": S[~S.is_exp].pin25.mean(), "treated_pin25": S[S.is_exp].pin25.mean(),
                         "placebo_pin10": S[~S.is_exp].pin10.mean(), "n_treated": int(S.is_exp.sum()),
                         "n_placebo": int((~S.is_exp).sum())}
# distance-bin x volatility dummies as regressors (19 x 3 - 1)
dv = pd.get_dummies(S.dv, prefix="dv", drop_first=True, dtype=float)
S = pd.concat([S, dv], axis=1)
DV = list(dv.columns)
for dep in ["pin25", "pin10", "toward_bp"]:
    # (a) original gamma-rank spec + distance x volatility controls
    run(res, f"A_{dep}_gamma_with_distance", dep, ["expx", "exp_gir"] + DV, S)
    # (b) OI-based intensity, slope in both groups, distance x volatility controls
    run(res, f"A_{dep}_oi_slopes", dep, ["expx", "oir", "exp_oir"] + DV, S)
    # (c) as (b) with date x size-decile FE
    run(res, f"A_{dep}_oi_slopes_size", dep, ["expx", "oir", "exp_oir"] + DV, S, extra_fe="size_dec")
# narrow window: starting distance between 25 and 100 bp only (outside the 0.25% target band)
N = S[(S.adist_prev > 25) & (S.adist_prev <= 100)]
run(res, "A_pin25_oi_slopes_25to100bp", "pin25", ["expx", "oir", "exp_oir"] + DV, N)

# ------------------------------------------------------------------ B. size and liquidity
X = ["expx", "exp_gir"] + CTRL
XP = ["expx", "exp_gir"] + CTRL_PRE
rb = run(res, "B_baseline_noearly", "abs_auc", X, F)
run(res, "B_predetermined_controls", "abs_auc", XP, F)
run(res, "B_no_controls", "abs_auc", ["expx", "exp_gir"], F)
run(res, "B_optionable_only", "abs_auc", X, F[F.optionable])
run(res, "B_date_x_size", "abs_auc", X, F, extra_fe="size_dec")
run(res, "B_optionable_date_x_size", "abs_auc", X, F[F.optionable], extra_fe="size_dec")
run(res, "B_date_x_liquidity", "abs_auc", X, F, extra_fe="liq_dec")
F["year"] = F.date.dt.year
share = F[F.optionable].groupby(["permno", "year"]).is_exp.mean().rename("sh_exp").reset_index()
F = F.merge(share, on=["permno", "year"], how="left")
run(res, "B_names_without_weeklies", "abs_auc", X, F[F.optionable & (F.sh_exp < 0.5)])

# ------------------------------------------------------------------ C. corrected magnitudes
b1, b2 = rb.params["expx"], rb.params["exp_gir"]
top = F[F.is_exp & (F.gir >= 0.8)]
eff = b1 + b2 * top.gir
res["C_pct_effect_top_quintile"] = {"mean_effect_bp": float(eff.mean()),
                                    "mean_observed_bp": float(top.abs_auc.mean()),
                                    "mean_counterfactual_bp": float((top.abs_auc - eff).mean()),
                                    "pct": float(eff.mean() / (top.abs_auc - eff).mean())}
print("C pct", res["C_pct_effect_top_quintile"], flush=True)

Rv = F[F.auc_bp.notna() & F.on_bp.notna()].copy()
Rv["auc"] = Rv.auc_bp
Rv["auc_x_exp"] = Rv.auc * Rv.expx
Rv["auc_x_exp_gir"] = Rv.auc * Rv.exp_gir
rr = run(res, "C_rev_auc", "on_bp", ["auc", "auc_x_exp", "auc_x_exp_gir", "expx", "exp_gir"], Rv)
V = rr.cov
tot = {}
for lab, w in {"non_expiring": [1, 0, 0], "expiring_gr0": [1, 1, 0], "expiring_gr1": [1, 1, 1]}.items():
    w = np.array(w + [0, 0], dtype=float)
    est = float(w @ rr.params.values)
    se = float(np.sqrt(w @ V.values @ w))
    tot[lab] = {"slope": est, "ci95": [est - 1.96 * se, est + 1.96 * se]}
res["C_total_reversal_slopes"] = tot
print("C reversal totals", tot, flush=True)
for k in list(res):
    if isinstance(res[k], dict) and "cov" in res[k]:
        res[k].pop("cov")
json.dump(res, open(OUT + "table6_robustness.json", "w"), indent=1, default=float)
print("done", flush=True)
