"""Relationship between the change in abstention and the change in bias across the
36 main-grid conditions (Results paragraph and Appendix correlation tables).

Reads results/tables/main_grid.csv and writes
  results/tables/abstention_correlation.csv   Spearman rho, p, and a 10,000-resample
                                              bootstrap CI over conditions, overall and per model
The per-condition values (A_base, A_SFT, delta A, delta s_AMB) are already in main_grid.csv.

Run: python analysis/abstention_correlation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES  # noqa: E402
from config import MODEL_ORDER  # noqa: E402

N_BOOT, SEED = 10_000, 42


def boot_ci(x, y):
    rng = np.random.default_rng(SEED)
    n = len(x)
    r = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return np.nanpercentile(r, 2.5), np.nanpercentile(r, 97.5)


def main():
    g = pd.read_csv(TABLES / "main_grid.csv")
    g["delta_abstention"] = g.abstention_ft - g.abstention_base
    out = []
    for name, d in [("all", g)] + [(m, g[g.model == m]) for m in MODEL_ORDER]:
        rho, p = spearmanr(d.delta_abstention, d.s_amb_delta)
        lo, hi = boot_ci(d.delta_abstention.values, d.s_amb_delta.values)
        out.append({"conditions": name, "n": len(d), "rho": rho, "p": p, "ci_low": lo, "ci_high": hi,
                    "abstention_decreases": int((d.delta_abstention < 0).sum()),
                    "mean_delta_abstention_pp": 100 * d.delta_abstention.mean()})
        print(f"{name:10s} n={len(d):2d} rho={rho:+.2f} p={p:.3f} CI=[{lo:+.2f}, {hi:+.2f}] "
              f"decreases={int((d.delta_abstention < 0).sum())} mean dA={100 * d.delta_abstention.mean():+.1f} pp")
    pd.DataFrame(out).to_csv(TABLES / "abstention_correlation.csv", index=False)


if __name__ == "__main__":
    main()
