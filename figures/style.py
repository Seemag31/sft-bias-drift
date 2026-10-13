"""Shared plotting conventions for all paper figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

BLACK = "#000000"
MODEL_COLOR = {"qwen": "#4285f4", "gemma": "#ea4335", "ministral": "#34a853"}
MODEL_LIGHT = {"qwen": "#c9dcfb", "gemma": "#f8cfcb", "ministral": "#c8e8d2"}
MODEL_NAME = {"qwen": "Qwen3.5-9B", "gemma": "Gemma-4-12B", "ministral": "Ministral-3-8B-Instruct"}
DOMAIN_COLOR = {"math": "#4285F4", "code": "#EA4335", "medical": "#FBBC04", "safety": "#34A853"}
DOMAIN_NAME = {"math": "Math", "code": "Code", "medical": "Medical", "safety": "Safety"}

CATEGORY_ORDER = ["Age", "SES", "Physical_appearance", "Disability_status", "Gender_identity",
                  "Nationality", "Sexual_orientation", "Race_ethnicity", "Religion"]
CATEGORY_LABEL = {"Age": "Age", "SES": "Socioeconomic\nstatus", "Physical_appearance": "Physical\nappearance",
                  "Disability_status": "Disability", "Gender_identity": "Gender\nidentity",
                  "Nationality": "Nationality", "Sexual_orientation": "Sexual\norientation",
                  "Race_ethnicity": "Race /\nethnicity", "Religion": "Religion"}
Y_FC = r"Fold change in $s_{\mathrm{AMB}}$"


def setup():
    plt.rcParams.update({"mathtext.default": "bf", "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": BLACK,
                         "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
                         "font.weight": "bold", "axes.labelweight": "bold"})
    FIGURES.mkdir(parents=True, exist_ok=True)


def bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def boxed_legend(ax, loc="upper right", size=11, **kw):
    leg = ax.legend(loc=loc, frameon=True, fancybox=False, framealpha=1.0, facecolor="white",
                    edgecolor="black", prop={"weight": "bold", "size": size}, handlelength=1.6,
                    borderpad=0.5, borderaxespad=0.4, **kw)
    leg.get_frame().set_linewidth(1.2)
    leg.set_zorder(6)
    return leg


def save(fig, name, dpi=300, **kw):
    fig.savefig(FIGURES / f"{name}.pdf", dpi=dpi, bbox_inches="tight", facecolor="white",
                metadata={"Creator": None, "Producer": None, "CreationDate": None}, **kw)
    print("wrote", name)
