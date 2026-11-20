# Option Expirations and the Closing Price

Essay for **21048 Market Structure and Information-Based Models** (Università Bocconi, a.y. 2026–2027).
Author: Luca Dennert.

Does delta hedging of expiring equity options distort the closing-auction price? Using CRSP and
OptionMetrics (2016 – Aug 2025), the essay measures hedging pressure as expiring dollar gamma relative to
normal dollar volume and compares stocks on the same day (stock and date fixed effects).

**Main results**

- High expiring gamma raises the probability of closing within 0.25% of the max-open-interest strike by
  4.5 pp (t = 8.3), against a placebo rate of 9.7%.
- Closing-price moves of high-gamma stocks are about 45% smaller than on their other Fridays.
- The attraction towards the strike does not reverse overnight.
- The 2022 Tuesday/Thursday SPX expirations have no robust effect on S&P 500 constituents' closes.

The paper is in [`paper/main.pdf`](paper/main.pdf).

## Repository structure

```
code/
  01_extract.py            raw data from WRDS (CRSP v2 daily, OptionMetrics IvyDB US, link tables)
  02_build_expiries.py     open interest by strike, Black-Scholes gamma, max-OI strike per expiration
  03_build_panel.py        stock-day panel: closing-move proxy, overnight return, gamma intensity
  04_checks.py             data checks (Appendix B)
  05_analysis.py           Tables 1-5 and data for Figure 1
  06_proxy_validation.py   CRSP closing quotes vs TAQ, 13 Feb 2009 (Appendix A)
  07_figure.py             Figure 1
results/                   estimation output (aggregate statistics only)
paper/                     LaTeX source, bibliography, figure and compiled PDF
```

## Replication

1. Run `code/01_extract.py` to `code/05_analysis.py` on the WRDS cloud (they read and write `~/essay_data/`).
   Step 1 takes about one hour; the rest a few minutes.
2. Copy `~/essay_data/results/*` into `results/`, then run `python code/07_figure.py`.
3. Compile the paper: `cd paper && latexmk -pdf main.tex`.

Requirements: Python 3 with `pandas`, `numpy`, `pyarrow`, `linearmodels`, `matplotlib`, `wrds`; a WRDS account
with access to CRSP and OptionMetrics.

## Data

CRSP, OptionMetrics and TAQ data are licensed through WRDS and are **not** included in this repository.
Only code and aggregate results are shared.
