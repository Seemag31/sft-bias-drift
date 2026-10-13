"""Figure: s_AMB before (open) and after (filled) fine-tuning for all 36 conditions,
rows grouped by domain, one line per model. Reads results/tables/table03_full_results.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import BLACK, MODEL_COLOR, TABLES, bold_ticks, save, setup  # noqa: E402
from config import MAIN_GRID  # noqa: E402

LABEL = {"openr1": "OpenR1", "nemotron": "Nemotron", "ultradata": "UltraData", "ocr": "OCR", "xcoder": "X-Coder",
         "starcoder_py": "Starcoder", "medo1": "Med-o1", "medcase": "MedCase", "mediflow": "MediFlow",
         "aegis": "Aegis", "wildjailbreak": "WildJailbrk", "pku_safe": "PKU"}
DLBL = {"math": "Math", "code": "Code", "medical": "Medical", "safety": "Safety"}
OFF = {"qwen": 0.26, "gemma": 0.0, "ministral": -0.26}
NAME = {"qwen": "Qwen", "gemma": "Gemma", "ministral": "Ministral"}


def main():
    setup()
    g = pd.read_csv(TABLES / "table03_full_results.csv").set_index(["model", "dataset"])
    rows = [(d, ds) for d, dsets in MAIN_GRID.items() for ds in dsets]
    N = len(rows); ys = np.arange(N)[::-1]
    fig, ax = plt.subplots(figsize=(11.8, 8.6), dpi=300)
    fig.subplots_adjust(left=0.20, bottom=0.13, top=0.97, right=0.98)
    for band_i, d0 in enumerate(range(0, N, 3)):
        if band_i % 2 == 1:
            ax.axhspan(ys[d0 + 2] - 0.5, ys[d0] + 0.5, color="#efefef", zorder=0)
    for d0 in range(3, N, 3):
        ax.axhline(ys[d0] + 0.5, color="#888888", lw=1.2, zorder=1)
    for k in range(1, N):                      # faint separators between datasets within a domain
        if k % 3:
            ax.axhline(ys[k] + 0.5, color="#bbbbbb", lw=0.8, ls=(0, (2, 3)), zorder=1)
    for (d, ds), y in zip(rows, ys):
        for m in ["qwen", "gemma", "ministral"]:
            b, f = g.loc[(m, ds), "s_amb_base"], g.loc[(m, ds), "s_amb_ft"]
            yy, c = y + OFF[m], MODEL_COLOR[m]
            ax.plot([b, f], [yy, yy], color=c, lw=2.6, zorder=3, solid_capstyle="round")
            ax.annotate("", xy=(f, yy), xytext=(b, yy),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0, mutation_scale=12), zorder=4)
            ax.scatter([b], [yy], s=62, facecolors="white", edgecolors=c, linewidths=2.0, zorder=5)
            ax.scatter([f], [yy], s=62, color=c, zorder=6)
    ax.axvline(0, color="#777777", lw=1.2, zorder=1)
    ax.set_ylim(-0.75, N - 0.25)
    ax.set_yticks(ys, [LABEL[ds] for _, ds in rows], fontsize=13.5)
    for d0 in range(0, N, 3):
        ax.annotate(DLBL[rows[d0][0]], xy=(-0.21, (ys[d0 + 1] + 0.5) / N), xycoords="axes fraction",
                    fontsize=15.5, fontweight="bold", ha="center", va="center", rotation=90, annotation_clip=False)
    ax.set_xlim(-0.055, 0.225)
    ax.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel(r"Ambiguous-context bias score  $s_{\mathrm{AMB}}$", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=13.5, length=0); bold_ticks(ax)
    ax.xaxis.grid(True, color="#dcdcdc", lw=0.9); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Line2D([], [], color=MODEL_COLOR[m], marker="o", ms=9, lw=3, label=NAME[m]) for m in ["qwen", "gemma", "ministral"]]
    handles += [Line2D([], [], marker="o", mfc="white", mec=BLACK, mew=2.0, ls="", ms=9, label="open = baseline"),
                Line2D([], [], marker="o", color=BLACK, ls="", ms=9, label="filled = after fine-tuning")]
    fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False, fontsize=13,
               bbox_to_anchor=(0.55, 0.005), columnspacing=1.6)
    save(fig, "fig06_bias_shift", dpi=400)


if __name__ == "__main__":
    main()
