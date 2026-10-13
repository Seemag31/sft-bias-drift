"""Invalid responses and decoding regime (Appendix F.1, Table 17).

Free generation lets the model reply however it likes; guided choice restricts the answer
to A, B or C. For the code and safety anchors (standard and abstention-augmented SFT) this
compares, per regime, the fold change in s_AMB and abstention, and reports the invalid-
response rate under free generation. Also reports the baseline and grid-wide invalid counts
quoted in the text.

Writes to results/tables/:
  table17_decoding_regime.csv   code and safety anchors x {standard, abstention-augmented}
  table17_invalid_summary.csv   baseline maximum, fine-tuned total and maximum, code gap

Run: python analysis/decoding_regime.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, bbq_metrics, load_predictions, paired  # noqa: E402
from config import MAIN_GRID, MAIN_REGIME, MODEL_ORDER  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def invalid_ambiguous(df):
    amb = df[df["context_condition"] == "ambig"]
    return int((amb["invalid"] | (amb["pred"] < 0)).sum()), len(amb)


def main():
    rows = []
    for m in MODEL_ORDER:
        for d in ("code", "safety"):
            for arm, run in (("standard", f"ext_{d}_s1"), ("augmented", f"ext_{d}_idk")):
                r = {"model": m, "domain": d, "sft": arm}
                for regime, suf in SUFFIX.items():
                    base, ft = paired(load_predictions(m, "baseline" + suf), load_predictions(m, run + suf))
                    mb, mf = bbq_metrics(base), bbq_metrics(ft)
                    r[f"fc_{regime}"] = mf["bias_s_amb"] / mb["bias_s_amb"]
                    r[f"abstention_{regime}"] = mf["abstention"]
                    r[f"invalid_{regime}"] = mf["invalid_rate"]
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table17_decoding_regime.csv", index=False)

    # counts quoted in F.1
    base_inv = [invalid_ambiguous(load_predictions(m, "baseline" + s))
                for m in MODEL_ORDER for s in SUFFIX.values()]
    ft_inv = [invalid_ambiguous(load_predictions(m, run + SUFFIX[MAIN_REGIME[d]]))
              for d, ds in MAIN_GRID.items() for run in ds.values() for m in MODEL_ORDER]
    code = t[t.domain == "code"]
    summary = {"baseline_max_invalid_rate": max(k / n for k, n in base_inv),
               "baseline_max_invalid_count": max(k for k, _ in base_inv),
               "finetuned_total_invalid": sum(k for k, _ in ft_inv),
               "finetuned_max_invalid_rate": max(k / n for k, n in ft_inv),
               "code_max_fc_gap_between_regimes": float((code.fc_free - code.fc_guided).abs().max())}
    pd.Series(summary).to_frame("value").to_csv(TABLES / "table17_invalid_summary.csv")
    print(t.round(3).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
