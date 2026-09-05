"""Experiment 2 category figure: pooled fold change per BBQ category under standard vs
abstention-augmented SFT over the 12 anchor pairs (pairs with an undefined baseline
excluded). Reads results/tables/per_category.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (BLACK, CATEGORY_LABEL, CATEGORY_ORDER, TABLES, Y_FC, bold_ticks,  # noqa: E402
                           boxed_legend, save, setup)
from config import ANCHORS, MAIN_REGIME  # noqa: E402

DARK, LIGHT = "#4285F4", "#c9dcfb"


def pooled_fc(pc, arm):
    runs = {f"ext_{d}_{arm}": MAIN_REGIME[d] for d in ANCHORS}
    d = pc[[runs.get(r) == g for r, g in zip(pc.run, pc.regime)]]
    d = d[d.s_amb_base.notna()]
    g = d.groupby("category").agg(base=("s_amb_base", "mean"), ft=("s_amb_ft", "mean"))
    return (g.ft / g.base).loc[CATEGORY_ORDER]


def main():
    setup()
    pc = pd.read_csv(TABLES / "per_category.csv")
    std, aug = pooled_fc(pc, "s1").values, pooled_fc(pc, "idk").values
    print(pd.DataFrame({"standard": std, "augmented": aug}, index=CATEGORY_ORDER).round(2).to_string())
    x = np.arange(len(CATEGORY_ORDER)); w = 0.38
    fig, ax = plt.subplots(figsize=(13.6, 6.2), dpi=300)
    ax.bar(x - w / 2, std, w, color=DARK, zorder=3, label="Standard SFT")
    ax.bar(x + w / 2, aug, w, color=LIGHT, edgecolor=DARK, hatch="xx", linewidth=1.2, zorder=3,
           label="Abstention-augmented SFT")
    ax.axhline(1, ls="--", color=BLACK, lw=2.0, zorder=2)
    ax.text(x[-1] + w + 0.1, 1.04, "no change", ha="right", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylabel(Y_FC, fontsize=21, fontweight="bold")
    ax.set_xticks(x, [CATEGORY_LABEL[c] for c in CATEGORY_ORDER], fontsize=18, fontweight="bold",
                  rotation=25, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=18, length=0); ax.tick_params(axis="x", length=0); bold_ticks(ax)
    ax.yaxis.grid(True, color="#cfcfcf", lw=1.1); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(2); ax.spines["bottom"].set_linewidth(2)
    ax.set_ylim(0, 1.08 * max(std.max(), aug.max()))
    boxed_legend(ax, size=17)
    ax.margins(x=0.01); fig.tight_layout(pad=0.3)
    save(fig, "Figure_abstention_category", dpi=400)


if __name__ == "__main__":
    main()
