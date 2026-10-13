"""Per-category results (Figures 3a and 7; Tables 5, 6, 12).

Scores every main-grid run and the abstention-augmented anchors on each BBQ category.
A baseline that abstains on every item in a category has s_AMB = 0, so the change is
defined for all 36 conditions in every category; only fold changes at a zero baseline
are excluded.

Writes to results/tables/:
  per_category.csv                   model x run x category: s_AMB before/after, change
  table05_06_category_changes.csv    mean change per category (overall and by domain) and
                                     the share of conditions with a positive change
  table12_category_fold_change.csv   pooled fold change per category, standard vs
                                     abstention-augmented SFT (12 anchor pairs)

Run: python analysis/per_category.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bbq_metrics, load_predictions, paired  # noqa: E402
from config import ANCHORS, BBQ_CATEGORIES, MAIN_GRID, MAIN_REGIME, MODEL_ORDER, RUNS  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}
GRID_RUNS = {(run, MAIN_REGIME[d]) for d, ds in MAIN_GRID.items() for run in ds.values()}
IDK_RUNS = {(f"ext_{d}_idk", MAIN_REGIME[d]) for d in ANCHORS}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in MODEL_ORDER:
        for run, regime in sorted(GRID_RUNS | IDK_RUNS):
            base, ft = paired(load_predictions(m, "baseline" + SUFFIX[regime]),
                              load_predictions(m, run + SUFFIX[regime]))
            for cat in BBQ_CATEGORIES:
                mb = bbq_metrics(base[base.category_file == cat])
                mf = bbq_metrics(ft[ft.category_file == cat])
                rows.append({"model": m, "domain": RUNS[run]["domain"], "run": run, "regime": regime,
                             "category": cat, "s_amb_base": mb["bias_s_amb"], "s_amb_ft": mf["bias_s_amb"],
                             "s_amb_delta": round(mf["bias_s_amb"] - mb["bias_s_amb"], 10)})
        print("[done]", m, flush=True)
    pc = pd.DataFrame(rows)
    pc.to_csv(TABLES / "per_category.csv", index=False)

    # Tables 5 and 6: the 36 standard-SFT conditions
    g = pc[[(r, rg) in GRID_RUNS for r, rg in zip(pc.run, pc.regime)]]
    t = g.groupby("category").s_amb_delta.agg(mean_delta="mean", n_conditions="size")
    t["positive_share"] = g.groupby("category").s_amb_delta.apply(lambda x: (x > 0).mean())
    t = t.join(g.pivot_table(index="category", columns="domain", values="s_amb_delta",
                             aggfunc="mean").add_prefix("mean_delta_")).reindex(BBQ_CATEGORIES)
    t.to_csv(TABLES / "table05_06_category_changes.csv")

    # Table 12: mean fine-tuned s_AMB over mean baseline s_AMB, per category
    fc = {}
    for name, arm in (("standard", "s1"), ("abstention_augmented", "idk")):
        keep = {(f"ext_{d}_{arm}", MAIN_REGIME[d]) for d in ANCHORS}
        a = pc[[(r, rg) in keep for r, rg in zip(pc.run, pc.regime)]]
        a = a[a.s_amb_base != 0]
        s = a.groupby("category").agg(b=("s_amb_base", "mean"), f=("s_amb_ft", "mean"))
        fc[name] = s.f / s.b
    pd.DataFrame(fc).reindex(BBQ_CATEGORIES).to_csv(TABLES / "table12_category_fold_change.csv")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
