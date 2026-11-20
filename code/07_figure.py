"""Step 7 - Figure 1 (pinning by gamma-intensity quintile).

Input: results/figure1_pinning_by_quintile.csv produced by 05_analysis.py.
Output: paper/figures/pinning_by_gamma.pdf
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "results", "figure1_pinning_by_quintile.csv"), index_col=0)
q = df.drop(index="placebo")
placebo = df.loc["placebo", "pin25"] * 100

labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
vals = q.pin25.values * 100
gi = q.gi_med.values * 100

fig, ax = plt.subplots(figsize=(7.2, 3.9))
colors = ["#bdbdb2"] * 4 + ["#2b6cb0"]
bars = ax.bar(range(5), vals, color=colors, width=0.6, zorder=2)
ax.hlines(placebo, -0.5, 4.35, color="#555555", ls="--", lw=1, zorder=3)
ax.text(4.42, placebo, f"Placebo\n{placebo:.1f}%", ha="left", va="center", fontsize=8.5, color="#555555")
ax.set_xlim(-0.5, 5.1)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(5))
ax.set_xticklabels([f"{l}\n{g:.2f}% of ADV" for l, g in zip(labels, gi)], fontsize=8.5)
ax.set_ylabel("Closes within 0.25% of $K^*$ (%)")
ax.set_ylim(0, 14.5)
ax.grid(axis="y", color="#e5e5e5", zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
out = os.path.join(HERE, "..", "paper", "figures", "pinning_by_gamma.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out)
print("saved", out)
