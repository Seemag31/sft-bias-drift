"""Demographic content of the training data vs bias change (Appendix C.6, Table 11).

For each of the 12 training files, the density of demographic terms per BBQ category
(lexicon hits per 10,000 words of prompt + response) is related to that category's change
in s_AMB. Density is log-scaled and standardized within category, the change within
model x category; the test is Spearman's rho, over all dataset x model x category
observations and over datasets (mean of models, with a dataset-level bootstrap CI).

Reads artifacts/train/<domain>_<variant>.jsonl and results/tables/per_category.csv.
Writes results/tables/table11_demographic_density.csv (one row per category, plus the
pooled test in results/tables/table11_pooled_test.csv).

Run: python analysis/demographic_content.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bbq_metrics import ARTIFACTS, TABLES  # noqa: E402
from config import BBQ_CATEGORIES, MAIN_GRID, MAIN_REGIME, RUNS  # noqa: E402

LEXICON = {
    "Age": "old older oldest elderly aged young younger youngest teen teens teenager teenagers senior seniors "
           "grandfather grandmother grandparent grandparents elder elders retired youth adolescent adolescents "
           "geriatric middle-aged",
    "Disability_status": "disabled disability disabilities wheelchair blind deaf autism autistic dyslexia handicapped "
                         "impairment impaired paralyzed paralysis schizophrenia bipolar adhd paraplegic amputee",
    "Gender_identity": "gender transgender nonbinary non-binary male female men women man woman boy girl boys girls "
                       "masculine feminine cisgender transsexual",
    "Nationality": "nationality nationalities american americans chinese indian indians mexican mexicans russian "
                   "russians german germans french japanese british african africans immigrant immigrants "
                   "foreigner foreigners citizen citizens",
    "Physical_appearance": "fat obese overweight thin skinny tall ugly beautiful attractive appearance bald handsome chubby",
    "Race_ethnicity": "black white asian asians hispanic hispanics latino latina latinos race racial racist ethnic "
                      "ethnicity caucasian arab arabs",
    "Religion": "christian christians muslim muslims jewish jew jews hindu hindus buddhist atheist atheists religion "
                "religious islam islamic church god bible quran mosque",
    "SES": "poor rich wealthy poverty income welfare homeless low-income affluent unemployed upper-class working-class",
    "Sexual_orientation": "gay lesbian bisexual homosexual lgbt lgbtq straight heterosexual queer homosexuality",
}
TERMS = {c: set(w.split()) for c, w in LEXICON.items()}
TOKEN = re.compile(r"[a-z][a-z'\-]*")
N_BOOT, SEED = 5000, 42


def density(path):
    """(words, lexicon hits per BBQ category) of one training file."""
    words, hits = 0, {c: 0 for c in TERMS}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        toks = TOKEN.findall((r["prompt"] + " " + r["response"]).lower())
        words += len(toks)
        for t in toks:
            for c, s in TERMS.items():
                hits[c] += t in s
    return words, hits


def zscore(x):
    return (x - x.mean()) / (x.std(ddof=0) or 1)


def main():
    pc = pd.read_csv(TABLES / "per_category.csv")
    obs, counts = [], {}
    for d, datasets in MAIN_GRID.items():
        for ds, run in datasets.items():
            words, hits = density(ARTIFACTS / "train" / f"{d}_{RUNS[run]['data_variant']}.jsonl")
            counts[(d, ds)] = (words, hits)
            s = pc[(pc.run == run) & (pc.regime == MAIN_REGIME[d])]
            for _, r in s.iterrows():
                obs.append({"domain": d, "dataset": ds, "model": r.model, "category": r.category,
                            "delta": r.s_amb_delta, "per10k": 1e4 * hits[r.category] / words})
    o = pd.DataFrame(obs)
    o["ld"] = np.log1p(o.per10k)
    o["ld_z"] = o.groupby("category").ld.transform(zscore)
    o["d_z"] = o.groupby(["model", "category"]).delta.transform(zscore)
    rho_all, p_all = stats.spearmanr(o.ld_z, o.d_z)

    a = o.groupby(["domain", "dataset", "category"], as_index=False).agg(
        delta=("delta", "mean"), ld=("ld", "first"), per10k=("per10k", "first"))
    a["ld_z"] = a.groupby("category").ld.transform(zscore)
    a["d_z"] = a.groupby("category").delta.transform(zscore)
    rho_ds, p_ds = stats.spearmanr(a.ld_z, a.d_z)
    rng = np.random.default_rng(SEED)
    keys = a[["domain", "dataset"]].drop_duplicates().values.tolist()
    boot = []
    for _ in range(N_BOOT):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        s = pd.concat([a[(a.domain == x) & (a.dataset == y)] for x, y in pick])
        if s.ld_z.std() and s.d_z.std():
            boot.append(stats.spearmanr(s.ld_z, s.d_z)[0])
    pooled = pd.DataFrame([
        {"test": "all observations", "n": len(o), "rho": rho_all, "p": p_all},
        {"test": "datasets (mean of models)", "n": len(a), "rho": rho_ds, "p": p_ds,
         "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)}])

    rows = []
    for c in BBQ_CATEGORIES:
        g = a[a.category == c]
        rho, p = stats.spearmanr(g.per10k, g.delta)
        pooled_density = 1e4 * sum(h[c] for _, h in counts.values()) / sum(w for w, _ in counts.values())
        rows.append({"category": c, "density_pooled": pooled_density,
                     "density_min": g.per10k.min(), "density_max": g.per10k.max(),
                     "mean_delta": g.delta.mean(), "rho": rho, "p": p})
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "table11_demographic_density.csv", index=False)
    pooled.to_csv(TABLES / "table11_pooled_test.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
