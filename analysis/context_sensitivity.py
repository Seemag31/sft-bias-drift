"""Context sensitivity of abstention training (Appendix D.4, Table 13).

Each BBQ scenario is an ambiguous item and its disambiguated twin. For the 12 anchor
conditions, standard SFT is compared with abstention-augmented SFT on the same scenarios:
  * abstention on ambiguous items, abstention and accuracy on disambiguated items;
  * desirable shift: answered the ambiguous item before, now abstains on it while
    answering its disambiguated twin correctly;
  * over-abstention: abstains on a disambiguated twin it previously answered correctly;
  * selectivity: change in ambiguous abstention minus change in disambiguated abstention.

Writes results/tables/table13_context_sensitivity.csv (all conditions and each model).

Run: python analysis/context_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, load_predictions, twin_frames  # noqa: E402
from config import ANCHORS, MAIN_REGIME, MODEL_ORDER  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def states(df):
    """Per-scenario outcome on the ambiguous item and on its disambiguated twin."""
    amb, dis = twin_frames(df)
    a_abs = (amb["pred"] == amb["unknown_loc"]).to_numpy()
    a_ans = ((amb["pred"] >= 0) & ~a_abs.astype(bool)).to_numpy()
    d_abs = (dis["pred"] == dis["unknown_loc"]).to_numpy()
    d_cor = dis["correct"].to_numpy().astype(bool) & ~d_abs
    return a_abs, a_ans, d_abs, d_cor


def main():
    rows = []
    for m in MODEL_ORDER:
        for d in ANCHORS:
            suf = SUFFIX[MAIN_REGIME[d]]
            sa_abs, sa_ans, sd_abs, sd_cor = states(load_predictions(m, f"ext_{d}_s1{suf}"))
            ia_abs, ia_ans, id_abs, id_cor = states(load_predictions(m, f"ext_{d}_idk{suf}"))
            rows.append({"model": m, "domain": d, "n": len(sa_abs),
                         "amb_abs_std": sa_abs.sum(), "amb_abs_aug": ia_abs.sum(),
                         "dis_abs_std": sd_abs.sum(), "dis_abs_aug": id_abs.sum(),
                         "dis_acc_std": sd_cor.sum(), "dis_acc_aug": id_cor.sum(),
                         "desirable": (sa_ans & ia_abs & id_cor).sum(), "over_abstention": (sd_cor & id_abs).sum()})
    c = pd.DataFrame(rows)

    def summarize(g, name):
        n = g.n.sum()
        pct = lambda col: 100 * g[col].sum() / n
        return {"group": name, "scenarios": int(n),
                "amb_abstention_std": pct("amb_abs_std"), "amb_abstention_aug": pct("amb_abs_aug"),
                "dis_abstention_std": pct("dis_abs_std"), "dis_abstention_aug": pct("dis_abs_aug"),
                "dis_accuracy_std": pct("dis_acc_std"), "dis_accuracy_aug": pct("dis_acc_aug"),
                "desirable_shift": pct("desirable"), "over_abstention": pct("over_abstention"),
                "selectivity_pp": (pct("amb_abs_aug") - pct("amb_abs_std")) - (pct("dis_abs_aug") - pct("dis_abs_std"))}

    out = pd.DataFrame([summarize(c, "all")] + [summarize(c[c.model == m], m) for m in MODEL_ORDER])
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "table13_context_sensitivity.csv", index=False)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
