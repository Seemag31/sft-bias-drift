"""Which group a model names when it commits on an ambiguous item (Results, Experiment 1).

For every ambiguous item where the model chose a non-unknown option, record the BBQ
group tag of that option (e.g. old/nonOld, lowSES/highSES) by category and question
polarity, before and after fine-tuning. `name_rate` = named / opportunities, where an
opportunity is an item on which that group appears as an option.

Writes results/tables/group_direction.csv (one row per model x run x category x
polarity x group, base and fine-tuned side by side).

Run: python analysis/group_direction.py [--runs ext_math_s1 ...]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, load_predictions, paired  # noqa: E402
from config import ANCHORS, MAIN_REGIME, MODEL_ORDER, RUNS  # noqa: E402


def cell_rows(df, model, run, arm):
    amb = df[df["context_condition"] == "ambig"]
    acc = defaultdict(lambda: {"opportunities": 0, "named": 0, "label": None})
    for _, r in amb.iterrows():
        info = json.loads(r["answer_info"])
        answers = json.loads(r["answers"])
        cat, pol, pred = r["category_file"], r["question_polarity"], int(r["pred"])
        for i in range(3):
            _, tag = info[f"ans{i}"]
            if str(tag).lower() == "unknown":
                continue
            a = acc[(cat, pol, tag)]
            a["opportunities"] += 1
            a["label"] = a["label"] or answers[i]
        if pred < 0:
            continue
        _, ptag = info[f"ans{pred}"]
        if str(ptag).lower() != "unknown":
            acc[(cat, pol, ptag)]["named"] += 1
    return [{"model": model, "run": run, "arm": arm, "category": cat, "polarity": pol, "group": tag,
             "example_answer": a["label"], "opportunities": a["opportunities"], "named": a["named"],
             "name_rate": a["named"] / a["opportunities"] if a["opportunities"] else float("nan")}
            for (cat, pol, tag), a in acc.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[f"ext_{d}_s1" for d in ANCHORS])
    args = ap.parse_args()
    rows = []
    for m in MODEL_ORDER:
        for run in args.runs:
            suf = "__guided" if MAIN_REGIME[RUNS[run]["domain"]] == "guided" else ""
            ft = load_predictions(m, run + suf)
            if ft is None:
                continue
            b, f = paired(load_predictions(m, "baseline" + suf), ft)
            rows += cell_rows(b, m, run, "base") + cell_rows(f, m, run, "ft")
            print(f"[ok] {m}/{run}", flush=True)
    long = pd.DataFrame(rows)
    key = ["model", "run", "category", "polarity", "group", "example_answer", "opportunities"]
    wide = long.pivot_table(index=key, columns="arm", values=["named", "name_rate"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide["name_rate_delta"] = wide["name_rate_ft"] - wide["name_rate_base"]
    wide = wide.sort_values(["model", "run", "category", "polarity", "name_rate_delta"],
                            ascending=[True, True, True, True, False])
    TABLES.mkdir(parents=True, exist_ok=True)
    wide.to_csv(TABLES / "group_direction.csv", index=False)
    print(f"group_direction.csv: {len(wide)} rows")


if __name__ == "__main__":
    main()
