"""Experiment 3 panels: fold change relative to each base model's direct-answer score on
the 1,100-item subset, horizontal bars (solid = direct, hatched = think then answer),
one panel per model with its own x-axis. Reads results/tables/deliberation_2x2.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import TABLES, Y_FC, bold_ticks, boxed_legend, save, setup  # noqa: E402

CONDS = ["baseline", "math", "code", "medical", "safety"]; LAB = ["Baseline", "Math", "Code", "Medical", "Safety"]
HUE = {"qwen": ("#00ACC1", "#C2EEF4"), "ministral": ("#FB8C00", "#FFE0B2"), "gemma": ("#8E24AA", "#E1BEE7")}


def main():
    setup()
    t = pd.read_csv(TABLES / "table15_deliberation.csv")
    for m in ["qwen", "ministral", "gemma"]:
        d = t[t.model == m].pivot(index="domain", columns="mode", values="fold_change_vs_base_direct").loc[CONDS]
        fd, ft = d["direct"].values, d["think"].values
        dark, light = HUE[m]
        fig, ax = plt.subplots(figsize=(6.6, 5.0)); y = np.arange(5)[::-1]; h = 0.36
        ax.barh(y + h / 2, fd, h, color=dark, zorder=3, label="Direct answer")
        ax.barh(y - h / 2, ft, h, color=light, edgecolor=dark, hatch="xx", linewidth=1.2, zorder=3,
                label="Think then answer")
        ax.axvline(1, ls="--", color="k", lw=2.5, zorder=2); ax.axvline(0, color="k", lw=2, zorder=2)
        lo = min(0, min(fd.min(), ft.min()) * 1.15); hi = max(2.2, max(fd.max(), ft.max()) * 1.28)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.75, 4.75)
        ax.text(1 + 0.012 * (hi - lo), -0.72, "no change", ha="left", va="bottom", fontsize=11, fontweight="bold",
                zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_yticks(y); ax.set_yticklabels(LAB, fontsize=16, fontweight="bold")
        ax.set_xlabel(Y_FC, fontsize=19, fontweight="bold")
        ax.tick_params(axis="x", labelsize=15); ax.tick_params(axis="y", length=0); bold_ticks(ax)
        ax.xaxis.grid(True, color="#d9d9d9", zorder=0); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(2); ax.spines["bottom"].set_linewidth(2)
        boxed_legend(ax)
        fig.tight_layout()
        save(fig, f"fig05_deliberation_{m}")


if __name__ == "__main__":
    main()
