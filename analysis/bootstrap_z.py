"""Paired-bootstrap standard errors and z statistics for the 36 main-grid cells (Table 2).

For each cell, items are resampled with replacement 1,000 times (seed 42), keeping each
item's base and fine-tuned answers together; SE is the standard deviation of the
resampled delta s_AMB and z = delta / SE.

Writes results/tables/standardized_drift.csv
Run: python analysis/bootstrap_z.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bootstrap_delta, load_predictions, paired  # noqa: E402
from config import MAIN_GRID, MAIN_REGIME, MODEL_ORDER  # noqa: E402

N_BOOT, SEED = 1000, 42


def main():
    rows = []
    for d, datasets in MAIN_GRID.items():
        suf = "__guided" if MAIN_REGIME[d] == "guided" else ""
        for ds, run in datasets.items():
            for m in MODEL_ORDER:
                ft = load_predictions(m, run + suf)
                if ft is None:
                    print(f"[skip] {m}/{run}{suf}")
                    continue
                b, f = paired(load_predictions(m, "baseline" + suf), ft)
                delta, se, z, ci = bootstrap_delta(b, f, N_BOOT, SEED)
                rows.append({"model": m, "domain": d, "dataset": ds, "run": run, "delta": delta,
                             "se": se, "z": z, "ci_low": ci[0], "ci_high": ci[1]})
                print(f"{m:9s} {ds:14s} delta={delta:+.4f} se={se:.4f} z={z:+.1f}", flush=True)
    t = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "standardized_drift.csv", index=False)
    print(f"significant increases (z>=2, delta>0): {int(((t.z >= 2) & (t.delta > 0)).sum())}/{len(t)}")


if __name__ == "__main__":
    main()
