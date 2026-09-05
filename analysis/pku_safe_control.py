"""PKU-SafeRLHF: safe-only responses (reported cell) vs "safer of the pair" responses
(control, 44% of targets carry an unsafe label), for each model and decoding regime.

Writes results/tables/pku_safe_control.csv with s_AMB before/after, delta, fold change,
bootstrap z (1,000 resamples, seed 42), and abstention.

Run: python analysis/pku_safe_control.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bbq_metrics, bootstrap_delta, load_predictions, paired  # noqa: E402
from config import MODEL_ORDER  # noqa: E402


def main():
    rows = []
    for m in MODEL_ORDER:
        for regime, suf in (("free", ""), ("guided", "__guided")):
            base = load_predictions(m, "baseline" + suf)
            for variant in ("pku_saferlhf", "pku_safe"):
                ft = load_predictions(m, f"ext_safety_{variant}{suf}")
                if ft is None:
                    continue
                b, f = paired(base, ft)
                mb, mf = bbq_metrics(b), bbq_metrics(f)
                delta, se, z, _ = bootstrap_delta(b, f)
                rows.append({"model": m, "variant": variant, "regime": regime, "n": len(b),
                             "s_amb_base": mb["bias_s_amb"], "s_amb_ft": mf["bias_s_amb"], "delta": delta,
                             "fold_change": mf["bias_s_amb"] / mb["bias_s_amb"], "se": se, "z": z,
                             "abstention_base": mb["abstention"], "abstention_ft": mf["abstention"],
                             "invalid_ft": mf["invalid_rate"]})
                print(f"{m:9s} {variant:13s} {regime:6s} FC={rows[-1]['fold_change']:.2f} z={z:+.1f}", flush=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "pku_safe_control.csv", index=False)


if __name__ == "__main__":
    main()
