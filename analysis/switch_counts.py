"""Item-level switch counts (Appendix C.4).

On the ambiguous age and socioeconomic-status items, counts the items the base model
answered with the non-committal option and the fine-tuned model answered with a
stereotype-aligned person, versus a stereotype-opposing person. Written for all 36
conditions; the paper quotes Qwen/OpenR1, Gemma/Med-o1 and Ministral/PKU-SafeRLHF.

Writes results/tables/c4_switch_counts.csv

Run: python analysis/switch_counts.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, load_predictions, paired, stereotype_choice  # noqa: E402
from config import MAIN_GRID, MAIN_REGIME, MODEL_ORDER  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def main():
    rows = []
    for d, datasets in MAIN_GRID.items():
        suf = SUFFIX[MAIN_REGIME[d]]
        for ds, run in datasets.items():
            for m in MODEL_ORDER:
                base, ft = paired(load_predictions(m, "baseline" + suf), load_predictions(m, run + suf))
                keep = ((base["context_condition"] == "ambig") & base["category_file"].isin(["Age", "SES"])).to_numpy()
                base, ft = base[keep], ft[keep]
                abstained = (base["pred"] == base["unknown_loc"]).to_numpy()
                stereo, anti = stereotype_choice(ft)
                rows.append({"model": m, "domain": d, "dataset": ds, "n_items": len(base),
                             "switch_to_stereotype": int((abstained & stereo).sum()),
                             "switch_to_anti_stereotype": int((abstained & anti).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "c4_switch_counts.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
