"""Experiment 3 (Table 15): s_AMB, accuracy, invalid rate, and abstention for every
{model} x {baseline, 4 anchor fine-tunes} x {direct, think} cell on the 1,100-item subset.

Reads artifacts/eval/deliberation/<model>/<run>/<mode>/predictions.parquet and writes
results/tables/table15_deliberation.csv (plus fold change relative to the base model's
direct-answer score, as plotted in the Experiment 3 figure).

Run: python analysis/deliberation_table.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import ARTIFACTS, TABLES, annotate_targets, bbq_metrics  # noqa: E402
from config import ANCHORS, MODEL_ORDER  # noqa: E402

DELIB = ARTIFACTS / "eval" / "deliberation"


def main():
    rows = []
    for m in MODEL_ORDER:
        for variant in ["baseline"] + [f"ext_{d}_s1" for d in ANCHORS]:
            for mode in ("direct", "think"):
                p = DELIB / m / variant / mode / "predictions.parquet"
                if not p.exists():
                    print(f"[skip] {p.relative_to(ROOT)}")
                    continue
                df = annotate_targets(pd.read_parquet(p))
                mt = bbq_metrics(df)
                rows.append({"model": m, "run": variant,
                             "domain": "baseline" if variant == "baseline" else variant.split("_")[1],
                             "mode": mode, "n": len(df), "s_amb": mt["bias_s_amb"], "accuracy": mt["accuracy"],
                             "invalid": mt["invalid_rate"], "abstention": mt["abstention"]})
    t = pd.DataFrame(rows)
    ref = t[(t.run == "baseline") & (t["mode"] == "direct")].set_index("model").s_amb
    t["fold_change_vs_base_direct"] = t.s_amb / t.model.map(ref)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "table15_deliberation.csv", index=False)
    print(t.pivot_table(index=["model", "domain"], columns="mode", values="s_amb").round(4).to_string())


if __name__ == "__main__":
    main()
