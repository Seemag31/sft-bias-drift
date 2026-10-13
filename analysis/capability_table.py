"""Target-task capability before and after fine-tuning (Table 15) from the benchmark evals.

Reads artifacts/eval/capability/<domain>/<model>/<run>/eval_meta.json and writes
results/tables/table15_capability.csv: standard (seed 1) and abstention-augmented anchors
per domain and model, with the baseline accuracy.

Run: python analysis/capability_table.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import ARTIFACTS, TABLES  # noqa: E402
from config import ANCHORS  # noqa: E402

CAP = ARTIFACTS / "eval" / "capability"


def main():
    rows = []
    for meta_path in sorted(CAP.glob("*/*/*/eval_meta.json")):
        m = json.loads(meta_path.read_text())
        rows.append({"domain": m["domain"], "model": m["model_key"], "run": m["variant"],
                     "benchmark": m["benchmark"], "n": m["n"], "accuracy": m["accuracy"]})
    t = pd.DataFrame(rows)
    base = t[t.run == "baseline"].set_index(["domain", "model"]).accuracy
    t["accuracy_base"] = [base.get((d, m)) for d, m in zip(t.domain, t.model)]
    t = t[t.run.isin([f"ext_{d}_{arm}" for d in ANCHORS for arm in ("s1", "idk")])]
    t = t.sort_values(["domain", "model", "run"])
    t["delta"] = t.accuracy - t.accuracy_base
    TABLES.mkdir(parents=True, exist_ok=True)
    t[["domain", "benchmark", "model", "run", "n", "accuracy_base", "accuracy", "delta"]].to_csv(
        TABLES / "table15_capability.csv", index=False)
    print(t.pivot_table(index=["domain", "model"], columns="run", values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
