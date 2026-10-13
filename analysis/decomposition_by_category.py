"""Bias decomposition by BBQ category (Appendix C.5, Table 10).

For each of the 36 conditions and each category, s_AMB = R * s_cond is split with the
midpoint rule used in decomposition.py:

    delta s_AMB = delta R * mean(s_cond)  +  mean(R) * delta s_cond

A category's contribution to the "more stereotyped answers" component is
mean(R) * delta s_cond averaged over the 36 conditions; its share is that contribution as a
percentage of the sum over categories (shares sum to 100; negative shares offset the total).
Shares are not weighted by category size.

Writes results/tables/table10_decomposition_category.csv

Run: python analysis/decomposition_by_category.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, load_predictions, stereotype_choice  # noqa: E402
from config import BBQ_CATEGORIES, MAIN_GRID, MAIN_REGIME, MODEL_ORDER  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def category_split(df):
    """{category: (R, s_cond)} on the ambiguous fold."""
    amb = df[df["context_condition"] == "ambig"]
    out = {}
    for c in BBQ_CATEGORIES:
        a = amb[amb["category_file"] == c]
        stereo, anti = stereotype_choice(a)
        n_bias, n_anti = int(stereo.sum()), int(anti.sum())
        out[c] = ((n_bias + n_anti) / len(a), (n_bias - n_anti) / max(n_bias + n_anti, 1))
    return out


def main():
    rows = []
    for d, datasets in MAIN_GRID.items():
        suf = SUFFIX[MAIN_REGIME[d]]
        for run in datasets.values():
            for m in MODEL_ORDER:
                base, ft = load_predictions(m, "baseline" + suf), load_predictions(m, run + suf)
                ci = base.index.intersection(ft.index)
                sb, sf = category_split(base.loc[ci]), category_split(ft.loc[ci])
                for c in BBQ_CATEGORIES:
                    (rb, cb), (rf, cf) = sb[c], sf[c]
                    rows.append({"category": c, "d_s_cond": cf - cb, "d_s_amb": rf * cf - rb * cb, "stereo_part": (rf + rb) / 2 * (cf - cb)})
    g = pd.DataFrame(rows).groupby("category")
    t = pd.DataFrame({"d_s_cond": g.d_s_cond.mean(), "d_s_amb": g.d_s_amb.mean(),
                      "stereo_contribution": g.stereo_part.mean()}).reindex(BBQ_CATEGORIES)
    t["share_pct"] = 100 * t.stereo_contribution / t.stereo_contribution.sum()
    t = t.sort_values("share_pct", ascending=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "table10_decomposition_category.csv")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
