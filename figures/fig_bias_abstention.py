"""Appendix scatter: change in abstention (percentage points) vs change in s_AMB over the
36 main-grid conditions, coloured by model. Reads results/tables/main_grid.csv."""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import MODEL_COLOR, TABLES, bold_ticks, boxed_legend, save, setup  # noqa: E402

NAME = {"qwen": "Qwen3.5-9B", "ministral": "Ministral-3-8B", "gemma": "Gemma-4-12B"}


def main():
    setup()
    g = pd.read_csv(TABLES / "main_grid.csv")
    g["dA"] = 100 * (g.abstention_ft - g.abstention_base)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axhline(0, ls="--", lw=1.5, color="k", zorder=2); ax.axvline(0, ls="--", lw=1.5, color="k", zorder=2)
    for m in ["qwen", "ministral", "gemma"]:
        d = g[g.model == m]
        ax.scatter(d.dA, d.s_amb_delta, s=70, color=MODEL_COLOR[m], edgecolor="black", linewidth=0.6,
                   zorder=3, label=NAME[m])
    # symmetric ranges on both axes so the two zero lines cross at the centre of the plot
    ylim = 1.08 * g.s_amb_delta.abs().max()
    xlim = 1.08 * g.dA.abs().max()
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel(r"$\Delta A$ (percentage points)", fontsize=17, fontweight="bold")
    ax.set_ylabel(r"$\Delta s_{\mathrm{AMB}}$", fontsize=17, fontweight="bold")
    ax.tick_params(labelsize=14); bold_ticks(ax)
    ax.grid(color="#d9d9d9", zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(2); ax.spines["bottom"].set_linewidth(2)
    boxed_legend(ax, loc="lower right", handletextpad=0.3, labelspacing=0.35, markerscale=0.9)
    fig.tight_layout()
    save(fig, "Figure_bias_abstention")


if __name__ == "__main__":
    main()
