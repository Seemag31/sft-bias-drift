"""Figure 3(a): pooled fold change in s_AMB by BBQ category and fine-tuning domain over the
36 main-grid cells (mean fine-tuned s_AMB over mean baseline s_AMB per category x domain;
cells with a zero baseline excluded). Reads results/tables/per_category.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (BLACK, CATEGORY_LABEL, CATEGORY_ORDER, DOMAIN_COLOR, DOMAIN_NAME,  # noqa: E402
                           TABLES, Y_FC, bold_ticks, save, setup)
from config import MAIN_GRID, MAIN_REGIME  # noqa: E402


def main():
    setup()
    pc = pd.read_csv(TABLES / "per_category.csv")
    keep = {(run, MAIN_REGIME[d]) for d, ds in MAIN_GRID.items() for run in ds.values()}
    pc = pc[[(r, g) in keep for r, g in zip(pc.run, pc.regime)]]
    pc = pc[pc.s_amb_base != 0]
    g = pc.groupby(["category", "domain"]).agg(base=("s_amb_base", "mean"), ft=("s_amb_ft", "mean"))
    fc = (g.ft / g.base).unstack("domain").loc[CATEGORY_ORDER]
    print(fc[["math", "code", "medical", "safety"]].round(2).to_string())
    x = np.arange(len(CATEGORY_ORDER)); w = 0.21
    fig, ax = plt.subplots(figsize=(13.6, 6.6), dpi=300)
    for i, d in enumerate(["math", "code", "medical", "safety"]):
        ax.bar(x + (i - 1.5) * w, fc[d].values, w, color=DOMAIN_COLOR[d], label=DOMAIN_NAME[d], zorder=3)
    ax.axhline(1, ls="--", color=BLACK, lw=2.0, zorder=2)
    ax.text(x[-1] + 2 * w, 1.04, "no change", ha="right", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylabel(Y_FC, fontsize=21, fontweight="bold")
    ax.set_xticks(x, [CATEGORY_LABEL[c] for c in CATEGORY_ORDER], fontsize=18, fontweight="bold",
                  rotation=25, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=18, length=0); ax.tick_params(axis="x", length=0); bold_ticks(ax)
    ax.yaxis.grid(True, color="#cfcfcf", lw=1.1); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(2); ax.spines["bottom"].set_linewidth(2)
    ax.set_ylim(0, 1.06 * fc[["math", "code", "medical", "safety"]].max().max())
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), ncols=4, frameon=False,
              prop={"weight": "bold", "size": 19}, columnspacing=1.2, handlelength=1.4)
    ax.margins(x=0.01); fig.tight_layout(pad=0.3)
    save(fig, "fig03a_category_domain", dpi=400)


if __name__ == "__main__":
    main()
