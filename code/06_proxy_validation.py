"""Step 6 - Validate the closing-quote proxy (Appendix A.1).

Compares CRSP closing bid/ask with TAQ quotes on 13 February 2009, the only day of millisecond TAQ
available in the WRDS sample libraries (taqmsamp). time_m is seconds after midnight; 57600 = 16:00.
"""
import pandas as pd
import wrds

db = wrds.Connection()

pre = db.raw_sql("""select distinct on (sym_root) sym_root, best_bid, best_ask from taqmsamp.nbbom_20090213
                    where sym_suffix is null and time_m < 57600 and best_bid > 0 and best_ask > 0
                    order by sym_root, time_m desc""")
post = db.raw_sql("""select distinct on (sym_root) sym_root, best_bid, best_ask from taqmsamp.nbbom_20090213
                     where sym_suffix is null and time_m between 57600 and 59400 and best_bid > 0 and best_ask > 0
                     order by sym_root, time_m desc""")
nyse = db.raw_sql("""select distinct on (sym_root) sym_root, bid, ask from taqmsamp.cqm_20090213
                     where sym_suffix is null and ex = 'N' and time_m < 57600 and bid > 0 and ask > 0
                     order by sym_root, time_m desc""")
cr = db.raw_sql("""select ticker, dlybid, dlyask, primaryexch from crsp.dsf_v2
                   where dlycaldt = '2009-02-13' and sharetype = 'NS' and securitytype = 'EQTY'""")

m = cr.merge(pre, left_on="ticker", right_on="sym_root").merge(post, on="sym_root", suffixes=("_pre", "_post"))
m = m[(m.dlybid > 0) & (m.dlyask > 0)].copy()
eq = lambda a, b, c, d: (abs(a - b) < 1e-6) & (abs(c - d) < 1e-6)
m["nbbo_pre"] = eq(m.dlybid, m.best_bid_pre, m.dlyask, m.best_ask_pre)
m["nbbo_post"] = eq(m.dlybid, m.best_bid_post, m.dlyask, m.best_ask_post)
print(m.groupby("primaryexch")[["nbbo_pre", "nbbo_post"]].agg(["mean", "count"]).round(3))

n = cr[cr.primaryexch == "N"].merge(nyse, left_on="ticker", right_on="sym_root")
n = n[n.dlybid > 0]
print("NYSE: equal to last NYSE quote before 16:00:", eq(n.dlybid, n.bid, n.dlyask, n.ask).mean().round(3), len(n))
