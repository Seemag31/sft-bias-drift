"""Figure 2: mean delta s_AMB by BBQ category and fine-tuning domain over the 36 main-grid
cells (cells with an undefined baseline excluded). Reads results/tables/per_category.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (BLACK, CATEGORY_LABEL, CATEGORY_ORDER, DOMAIN_COLOR, DOMAIN_NAME,  # noqa: E402
                           TABLES, bold_ticks, save, setup)
from config import MAIN_GRID, MAIN_REGIME  # noqa: E402


def main_grid_rows(pc):
    keep = {(run, MAIN_REGIME[d]) for d, dsets in MAIN_GRID.items() for run in dsets.values()}
    return pc[[(r, g) in keep for r, g in zip(pc.run, pc.regime)]]


def main():
    setup()
    pc = main_grid_rows(pd.read_csv(TABLES / "per_category.csv"))
    pc = pc[pc.s_amb_delta.notna()]
    mean = pc.groupby(["category", "domain"]).s_amb_delta.mean().unstack("domain").loc[CATEGORY_ORDER]
    x = np.arange(len(CATEGORY_ORDER)); w = 0.21
    fig, ax = plt.subplots(figsize=(13.6, 6.6), dpi=300)
    for i, d in enumerate(["math", "code", "medical", "safety"]):
        ax.bar(x + (i - 1.5) * w, mean[d].values, w, color=DOMAIN_COLOR[d], label=DOMAIN_NAME[d], zorder=3)
    ax.axhline(0, color=BLACK, lw=2.0, zorder=2)
    ax.set_ylabel(r"$\Delta s_{\mathrm{AMB}}$  (fine-tuned $-$ base)", fontsize=21, fontweight="bold")
    ax.set_xticks(x, [CATEGORY_LABEL[c] for c in CATEGORY_ORDER], fontsize=18, fontweight="bold",
                  rotation=25, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=18, length=0); ax.tick_params(axis="x", length=0); bold_ticks(ax)
    ax.yaxis.grid(True, color="#cfcfcf", lw=1.1); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), ncols=4, frameon=False,
              prop={"weight": "bold", "size": 19}, columnspacing=1.2, handlelength=1.4)
    ax.margins(x=0.01); fig.tight_layout(pad=0.3)
    save(fig, "Figure2_category_domain_bars", dpi=400)
    print(mean.round(3).to_string())


if __name__ == "__main__":
    main()
