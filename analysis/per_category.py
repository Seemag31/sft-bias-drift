"""Per-category s_AMB before/after fine-tuning for every run and regime.

Feeds the category tables and figures (Figure 2, Appendix category tables,
Experiment 2 category figure). Cells whose baseline has no committed answers
(Gemma on disability, gender identity, sexual orientation) are left NaN.

Writes results/tables/per_category.csv
Run: python analysis/per_category.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bbq_metrics, load_predictions, paired  # noqa: E402
from config import BBQ_CATEGORIES, MODEL_ORDER, RUNS  # noqa: E402


def main():
    rows = []
    for m in MODEL_ORDER:
        for regime, suf in (("free", ""), ("guided", "__guided")):
            base = load_predictions(m, "baseline" + suf)
            if base is None:
                continue
            for run, spec in RUNS.items():
                ft = load_predictions(m, run + suf)
                if ft is None:
                    continue
                b, f = paired(base, ft)
                for cat in BBQ_CATEGORIES:
                    bs, fs = b[b.category_file == cat], f[f.category_file == cat]
                    mb, mf = bbq_metrics(bs), bbq_metrics(fs)
                    rows.append({"model": m, "domain": spec["domain"], "dataset": spec["dataset"], "run": run,
                                 "regime": regime, "category": cat, "n": len(bs),
                                 "s_amb_base": mb["bias_s_amb"], "s_amb_ft": mf["bias_s_amb"],
                                 "s_amb_delta": mf["bias_s_amb"] - mb["bias_s_amb"],
                                 "abstention_ft": mf["abstention"], "invalid_ft": mf["invalid_rate"]})
            print(f"[done] {m}/{regime}", flush=True)
    t = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "per_category.csv", index=False)
    print(f"per_category.csv: {len(t)} rows")


if __name__ == "__main__":
    main()
