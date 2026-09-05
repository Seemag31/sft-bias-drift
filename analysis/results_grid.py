"""Headline tables from the BBQ predictions (Experiments 1 and 2).

Writes to results/tables/:
  results_grid.csv        every model x run x decoding regime: s_AMB before/after,
                          delta, abstention, invalid rate, accuracy
  main_grid.csv           the 36 main-grid cells (12 datasets x 3 models) in the
                          paper's regime (free for math/medical, guided for code/safety)
  seed_replication.csv    seed 1 vs seed 2 for the 4 anchor datasets
  abstention_control.csv  standard vs abstention-augmented SFT, 12 anchor pairs

Run: python analysis/results_grid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bbq_metrics, load_predictions, paired  # noqa: E402
from config import ANCHORS, MAIN_GRID, MAIN_REGIME, MODEL_ORDER, RUNS  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def row(model, run, regime):
    ft = load_predictions(model, run + SUFFIX[regime])
    if ft is None:
        return None
    base = load_predictions(model, "baseline" + SUFFIX[regime])
    b, f = paired(base, ft)
    mb, mf = bbq_metrics(b), bbq_metrics(f)
    return {"model": model, "run": run, "domain": RUNS[run]["domain"], "dataset": RUNS[run]["dataset"],
            "regime": regime, "n": len(b),
            "s_amb_base": mb["bias_s_amb"], "s_amb_ft": mf["bias_s_amb"],
            "s_amb_delta": mf["bias_s_amb"] - mb["bias_s_amb"],
            "fold_change": mf["bias_s_amb"] / mb["bias_s_amb"] if mb["bias_s_amb"] else np.nan,
            "abstention_base": mb["abstention"], "abstention_ft": mf["abstention"],
            "invalid_ft": mf["invalid_rate"], "acc_base": mb["accuracy"], "acc_ft": mf["accuracy"]}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = [r for m in MODEL_ORDER for run in RUNS for reg in ("free", "guided")
            if (r := row(m, run, reg)) is not None]
    grid = pd.DataFrame(rows)
    grid.to_csv(TABLES / "results_grid.csv", index=False)
    print(f"results_grid.csv: {len(grid)} rows")

    # ---- the 36 main-grid cells
    cells = []
    for d, datasets in MAIN_GRID.items():
        for ds, run in datasets.items():
            cells += [r for r in rows if r["run"] == run and r["regime"] == MAIN_REGIME[d]]
    main = pd.DataFrame(cells)[["model", "domain", "dataset", "run", "regime", "s_amb_base", "s_amb_ft",
                                "s_amb_delta", "fold_change", "abstention_base", "abstention_ft", "invalid_ft"]]
    main.to_csv(TABLES / "main_grid.csv", index=False)
    print(f"main_grid.csv: {len(main)} cells | FC>1 in {int((main.fold_change > 1).sum())}")

    # ---- seed replication (anchors, main regime)
    rep = []
    for m in MODEL_ORDER:
        for d in ANCHORS:
            reg = MAIN_REGIME[d]
            r1 = [r for r in rows if r["model"] == m and r["run"] == f"ext_{d}_s1" and r["regime"] == reg]
            r2 = [r for r in rows if r["model"] == m and r["run"] == f"ext_{d}_s2" and r["regime"] == reg]
            if r1 and r2:
                d1, d2 = r1[0]["s_amb_delta"], r2[0]["s_amb_delta"]
                rep.append({"model": m, "domain": d, "regime": reg, "delta_seed1": d1, "delta_seed2": d2,
                            "same_sign": np.sign(d1) == np.sign(d2), "abs_gap": abs(d1 - d2)})
    pd.DataFrame(rep).to_csv(TABLES / "seed_replication.csv", index=False)

    # ---- abstention-augmented control (Experiment 2)
    ctl = []
    for m in MODEL_ORDER:
        for d in ANCHORS:
            reg = MAIN_REGIME[d]
            n_ = [r for r in rows if r["model"] == m and r["run"] == f"ext_{d}_s1" and r["regime"] == reg]
            i_ = [r for r in rows if r["model"] == m and r["run"] == f"ext_{d}_idk" and r["regime"] == reg]
            if n_ and i_:
                n_, i_ = n_[0], i_[0]
                ctl.append({"model": m, "domain": d, "regime": reg,
                            "s_amb_base": n_["s_amb_base"],
                            "delta_standard": n_["s_amb_delta"], "delta_augmented": i_["s_amb_delta"],
                            "fc_standard": n_["fold_change"], "fc_augmented": i_["fold_change"],
                            "abstention_standard": n_["abstention_ft"], "abstention_augmented": i_["abstention_ft"]})
    pd.DataFrame(ctl).to_csv(TABLES / "abstention_control.csv", index=False)
    print(f"seed_replication.csv: {len(rep)} | abstention_control.csv: {len(ctl)}")


if __name__ == "__main__":
    main()
