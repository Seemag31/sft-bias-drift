"""Figure 7: pooled fold change per BBQ category under standard vs abstention-augmented SFT
over the 12 anchor pairs. Reads results/tables/table11_category_fold_change.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (BLACK, CATEGORY_LABEL, CATEGORY_ORDER, TABLES, Y_FC, bold_ticks,  # noqa: E402
                           boxed_legend, save, setup)

DARK, LIGHT = "#4285F4", "#c9dcfb"


def main():
    setup()
    t = pd.read_csv(TABLES / "table11_category_fold_change.csv", index_col="category").loc[CATEGORY_ORDER]
    std, aug = t["standard"].values, t["abstention_augmented"].values
    print(t.round(2).to_string())
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
    save(fig, "fig07_category_abstention", dpi=400)


if __name__ == "__main__":
    main()
