"""Experiments 1 and 2 headline tables from the BBQ predictions.

Writes to results/tables/:
  table03_full_results.csv        the 36 conditions (also Tables 1 and 8): s_AMB before/after,
                                  change, fold change, bootstrap z, abstention, invalid rate
  table04_seed_replication.csv    seed 1 vs seed 2 for the 12 anchor conditions
  table13_abstention_control.csv  standard vs abstention-augmented SFT, 12 anchor pairs

Standard errors come from a paired item bootstrap (1,000 resamples, seed 42):
z = delta s_AMB / SE.

Run: python analysis/main_results.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bbq_metrics, bootstrap_delta, load_predictions, paired  # noqa: E402
from config import ANCHORS, MAIN_GRID, MAIN_REGIME, MODEL_ORDER, RUNS  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def cell(model, run, regime, with_z=False):
    base, ft = paired(load_predictions(model, "baseline" + SUFFIX[regime]),
                      load_predictions(model, run + SUFFIX[regime]))
    mb, mf = bbq_metrics(base), bbq_metrics(ft)
    out = {"model": model, "domain": RUNS[run]["domain"], "dataset": RUNS[run]["dataset"], "run": run,
           "regime": regime, "s_amb_base": mb["bias_s_amb"], "s_amb_ft": mf["bias_s_amb"],
           "s_amb_delta": round(mf["bias_s_amb"] - mb["bias_s_amb"], 10),
           "fold_change": mf["bias_s_amb"] / mb["bias_s_amb"] if mb["bias_s_amb"] else np.nan,
           "abstention_base": mb["abstention"], "abstention_ft": mf["abstention"],
           "delta_abstention": mf["abstention"] - mb["abstention"], "invalid_ft": mf["invalid_rate"]}
    if with_z:
        _, se, z, _ = bootstrap_delta(base, ft)
        out.update({"se": se, "z": z})
    return out


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    # Tables 1, 3, 8: the 36 main-grid conditions
    grid = pd.DataFrame([cell(m, run, MAIN_REGIME[d], with_z=True)
                         for d, datasets in MAIN_GRID.items() for run in datasets.values()
                         for m in MODEL_ORDER])
    grid.to_csv(TABLES / "table03_full_results.csv", index=False)
    sig = (grid.z >= 2) & (grid.s_amb_delta > 0)
    print(f"table03: {len(grid)} conditions | substantial increases {int(sig.sum())} | "
          f"any increase {int((grid.s_amb_delta > 0).sum())} | mean FC {grid.fold_change.mean():.2f}")

    # Table 4: second seed (anchors)
    seed = []
    for m in MODEL_ORDER:
        for d in ANCHORS:
            a, b = cell(m, f"ext_{d}_s1", MAIN_REGIME[d]), cell(m, f"ext_{d}_s2", MAIN_REGIME[d])
            seed.append({"model": m, "domain": d, "delta_seed1": a["s_amb_delta"],
                         "delta_seed2": b["s_amb_delta"],
                         "same_sign": bool(np.sign(a["s_amb_delta"]) == np.sign(b["s_amb_delta"])),
                         "abs_gap": abs(a["s_amb_delta"] - b["s_amb_delta"])})
    seed = pd.DataFrame(seed)
    seed.to_csv(TABLES / "table04_seed_replication.csv", index=False)
    print(f"table04: direction matches {int(seed.same_sign.sum())}/12, median gap {seed.abs_gap.median():.4f}")

    # Table 13: standard vs abstention-augmented SFT
    ctl = []
    for m in MODEL_ORDER:
        for d in ANCHORS:
            n, i = cell(m, f"ext_{d}_s1", MAIN_REGIME[d]), cell(m, f"ext_{d}_idk", MAIN_REGIME[d])
            ctl.append({"model": m, "domain": d, "regime": MAIN_REGIME[d], "s_amb_base": n["s_amb_base"],
                        "delta_standard": n["s_amb_delta"], "delta_augmented": i["s_amb_delta"],
                        "fc_standard": n["fold_change"], "fc_augmented": i["fold_change"],
                        "abstention_standard": n["abstention_ft"], "abstention_augmented": i["abstention_ft"]})
    ctl = pd.DataFrame(ctl)
    ctl.to_csv(TABLES / "table13_abstention_control.csv", index=False)
    up = ctl.fc_standard > 1
    print(f"table13: augmentation lowers FC in {int((ctl.fc_augmented < ctl.fc_standard).sum())}/12; "
          f"in {int((ctl.fc_augmented[up] < ctl.fc_standard[up]).sum())}/{int(up.sum())} conditions "
          f"where standard SFT increases bias")


if __name__ == "__main__":
    main()
