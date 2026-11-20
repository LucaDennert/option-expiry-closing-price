"""Step 1 - Extract raw data from WRDS (CRSP v2 daily, OptionMetrics IvyDB US, link tables).

Run on the WRDS cloud (JupyterHub or SSH). Output: parquet files in ~/essay_data/raw/.
Raw WRDS data are licensed and are NOT included in this repository.
"""
import os
import time
import traceback

import pandas as pd
import wrds

OUT = os.path.expanduser("~/essay_data/raw")
LOG = os.path.expanduser("~/essay_data/extract.log")
WRDS_USER = os.environ.get("WRDS_USERNAME")  # set your WRDS username, or wrds will prompt for it
YEARS_CRSP = range(2015, 2026)
YEARS_OM = range(2016, 2026)  # OptionMetrics on WRDS ends 2025-08-29


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def save(df, name):
    df.to_parquet(os.path.join(OUT, name + ".parquet"), index=False)
    log(f"saved {name} rows={len(df)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    db = wrds.Connection(wrds_username=WRDS_USER)

    # static / link tables
    save(db.raw_sql("select * from optionm.securd"), "om_securd")
    save(db.raw_sql("select * from optionm.secnmd"), "om_secnmd")
    save(db.raw_sql("select * from wrdsapps.opcrsphist"), "link_om_crsp")
    save(db.raw_sql("select * from crsp.dsp500list_v2"), "crsp_sp500list")

    # CRSP v2 daily: ordinary common shares of US-incorporated firms on NYSE / Nasdaq / NYSE American
    for y in YEARS_CRSP:
        q = f"""select permno, dlycaldt as date, ticker, primaryexch, siccd, dlyprc, dlyopen, dlyhigh, dlylow,
                       dlyclose, dlybid, dlyask, dlyvol, dlynumtrd, dlyret, dlyretx, dlycap, shrout,
                       dlycumfacpr, dlyprcflg
                from crsp.dsf_v2
                where dlycaldt between '{y}-01-01' and '{y}-12-31'
                  and sharetype='NS' and securitytype='EQTY' and securitysubtype='COM'
                  and usincflg='Y' and primaryexch in ('N','Q','A')"""
        save(db.raw_sql(q), f"crsp_dsf_{y}")

    for y in YEARS_OM:
        # contracts 0-1 days from expiration, aggregated by underlying, date, expiry, strike and call/put
        q1 = f"""select secid, date, exdate, strike_price/1000.0 as k, cp_flag,
                        max(expiry_indicator) as ei, max(am_settlement) as am,
                        sum(open_interest) oi, sum(open_interest*gamma*contract_size) goi,
                        sum(volume) vol, avg(impl_volatility) iv, avg(delta) delta
                 from optionm.opprcd{y}
                 where exdate - date between 0 and 1 and open_interest > 0
                 group by 1,2,3,4,5"""
        save(db.raw_sql(q1), f"om_nearexp_strikes_{y}")

        # all options, aggregated by underlying and day
        q2 = f"""select secid, date, sum(open_interest) oi_all, sum(open_interest*gamma*contract_size) goi_all,
                        sum(volume) vol_all, count(distinct exdate) n_exp
                 from optionm.opprcd{y} group by 1,2"""
        save(db.raw_sql(q2), f"om_secday_{y}")

        # expiration calendar
        q3 = f"""select exdate, max(expiry_indicator) ei, count(distinct secid) n_secid
                 from optionm.opprcd{y} group by 1"""
        save(db.raw_sql(q3), f"om_expiries_{y}")

        # underlying prices
        save(db.raw_sql(f"select * from optionm.secprd{y}"), f"om_secprd_{y}")

        # placebo grid: on every Friday, strikes and open interest of the NEXT expiration
        q4 = f"""with x as (select secid, date, exdate, strike_price/1000.0 k, cp_flag, open_interest oi,
                                   min(exdate) over (partition by secid, date) mn
                            from optionm.opprcd{y}
                            where extract(isodow from date)=5 and exdate - date between 1 and 35
                              and open_interest > 0)
                 select secid, date, exdate, k,
                        sum(case when cp_flag='C' then oi else 0 end) oi_c,
                        sum(case when cp_flag='P' then oi else 0 end) oi_p
                 from x where exdate = mn group by 1,2,3,4"""
        save(db.raw_sql(q4), f"om_friday_nextgrid_{y}")
    log("ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # keep a trace in the log when run with nohup
        log("ERROR " + repr(e) + " " + traceback.format_exc()[-800:])
        raise
