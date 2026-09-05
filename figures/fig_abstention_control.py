"""Experiment 2 panels: fold change under standard (solid) vs abstention-augmented (hatched)
SFT, one panel per model. Qwen and Ministral share a y-axis; Gemma has its own.
Reads results/tables/abstention_control.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import MODEL_COLOR, MODEL_LIGHT, TABLES, Y_FC, bold_ticks, boxed_legend, save, setup  # noqa: E402

DOMS = ["math", "code", "medical", "safety"]; LAB = ["Math", "Code", "Medical", "Safety"]


def main():
    setup()
    t = pd.read_csv(TABLES / "abstention_control.csv")
    shared_max = 1.05 * t[t.model.isin(["qwen", "ministral"])][["fc_standard", "fc_augmented"]].max().max()
    for m in ["qwen", "ministral", "gemma"]:
        d = t[t.model == m].set_index("domain").loc[DOMS]
        fig, ax = plt.subplots(figsize=(6.2, 5.3))
        x = np.arange(4); w = 0.36
        ax.bar(x - w / 2, d.fc_standard, w, color=MODEL_COLOR[m], zorder=3, label="Standard SFT")
        ax.bar(x + w / 2, d.fc_augmented, w, color=MODEL_LIGHT[m], edgecolor=MODEL_COLOR[m], hatch="xx",
               linewidth=1.2, zorder=3, label="Abstention-augmented SFT")
        boxed_legend(ax)
        ax.axhline(1, ls="--", color="k", lw=2.5, zorder=2)
        ymax = shared_max if m != "gemma" else 1.22 * d[["fc_standard", "fc_augmented"]].max().max()
        ax.set_ylim(0, ymax)
        ax.text(3.9, 1 + 0.015 * ymax, "no change", ha="right", va="bottom", fontsize=11, fontweight="bold",
                zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=16, fontweight="bold")
        ax.set_ylabel(Y_FC, fontsize=19, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18); bold_ticks(ax)
        ax.set_xlim(-0.6, 3.95); ax.grid(axis="y", color="#d9d9d9", zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(2); ax.spines["bottom"].set_linewidth(2)
        fig.tight_layout()
        save(fig, f"Figure_abstention_control_FC_{m}")


if __name__ == "__main__":
    main()
