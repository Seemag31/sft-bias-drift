"""Bias decomposition and the disambiguated-fold check (Appendix C.5, Table 9).

(a) s_AMB = R * s_cond, with R = (n_bias + n_anti) / N the share of ambiguous items that
    receive a valid, non-abstaining answer and s_cond = (n_bias - n_anti) / (n_bias + n_anti)
    the stereotype preference among answered items. A midpoint split makes the two parts
    sum exactly to the change:  delta s_AMB = delta R * mean(s_cond) + mean(R) * delta s_cond.
(b) Disambiguated twins of the same items: accuracy, stereotype-aligned errors, accuracy
    after a stereotypical ambiguous answer, and the rate of following contradicting evidence.

Reads the 36 main-grid conditions; writes results/tables/table09_decomposition.csv
(rows: all conditions and each model).

Run: python analysis/decomposition.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import TABLES, load_predictions, stereotype_choice, twin_frames  # noqa: E402
from config import MAIN_GRID, MAIN_REGIME, MODEL_ORDER  # noqa: E402

SUFFIX = {"free": "", "guided": "__guided"}


def split(df):
    """R and s_cond on the ambiguous fold."""
    amb = df[df["context_condition"] == "ambig"]
    stereo, anti = stereotype_choice(amb)
    wrong = ~amb["correct"].to_numpy()
    n_bias, n_anti = int((stereo & wrong).sum()), int((anti & wrong).sum())
    n = len(amb)
    return (n_bias + n_anti) / n, (n_bias - n_anti) / max(n_bias + n_anti, 1)


def disambiguated(base, ft):
    """Per-cell quantities on the disambiguated twins (all counts are over items)."""
    out = {}
    for tag, df in (("base", base), ("sft", ft)):
        amb, dis = twin_frames(df)
        correct = dis["correct"].to_numpy()
        s_dis, _ = stereotype_choice(dis)
        out[f"{tag}_acc"] = correct.mean()
        out[f"{tag}_err_stereo"] = (~correct & s_dis).mean()
        if tag == "sft":
            guessed, _ = stereotype_choice(amb)                       # stereotypical ambiguous answer
            t, u = dis["target_loc"].to_numpy(dtype=float), dis["unknown_loc"].to_numpy(dtype=float)
            lab, neg = dis["label"].to_numpy(), (dis["question_polarity"] == "neg").to_numpy()
            known = ~np.isnan(t) & ~np.isnan(u)
            evidence_is_stereo = known & np.where(neg, lab == t, lab != t)
            conflict = guessed & known & ~evidence_is_stereo          # evidence contradicts the guess
            out["n_guessed"], out["n_after_guess_correct"] = int(guessed.sum()), int(correct[guessed].sum())
            out["n_conflict"], out["n_conflict_correct"] = int(conflict.sum()), int(correct[conflict].sum())
    return out


def main():
    rows = []
    for d, datasets in MAIN_GRID.items():
        suf = SUFFIX[MAIN_REGIME[d]]
        for run in datasets.values():
            for m in MODEL_ORDER:
                base, ft = load_predictions(m, "baseline" + suf), load_predictions(m, run + suf)
                ci = base.index.intersection(ft.index)
                base, ft = base.loc[ci], ft.loc[ci]
                (rb, sb), (rf, sf) = split(base), split(ft)
                dr, ds = rf - rb, sf - sb
                rows.append({"model": m, "d_s_amb": rf * sf - rb * sb, "d_R": dr, "d_s_cond": ds,
                             "part_answer": dr * (sf + sb) / 2, "part_stereo": (rf + rb) / 2 * ds,
                             **disambiguated(base, ft)})
        print("[done]", d, flush=True)
    cells = pd.DataFrame(rows)

    def summarize(g, name):
        total = g.d_s_amb.sum()
        return {"group": name, "n_conditions": len(g), "mean_d_s_amb": g.d_s_amb.mean(),
                "mean_d_R_pp": 100 * g.d_R.mean(), "mean_d_s_cond": g.d_s_cond.mean(),
                "share_answer_more": g.part_answer.sum() / total, "share_stereo": g.part_stereo.sum() / total,
                "cells_s_amb_up": int((g.d_s_amb > 0).sum()), "cells_R_up": int((g.d_R > 0).sum()),
                "cells_s_cond_up": int((g.d_s_cond > 0).sum()),
                "acc_base": g.base_acc.mean(), "acc_sft": g.sft_acc.mean(), "cells_acc_up": int((g.sft_acc > g.base_acc).sum()),
                "stereo_err_base": g.base_err_stereo.mean(), "stereo_err_sft": g.sft_err_stereo.mean(),
                "correct_after_stereo_guess": g.n_after_guess_correct.sum() / g.n_guessed.sum(),
                "follows_contradicting_evidence": g.n_conflict_correct.sum() / g.n_conflict.sum()}

    out = pd.DataFrame([summarize(cells, "all")] + [summarize(cells[cells.model == m], m) for m in MODEL_ORDER])
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "table09_decomposition.csv", index=False)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    main()
