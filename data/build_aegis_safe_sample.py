"""Aegis 2.0 training sample: safe-labelled responses only.

Keeps the rows of the Aegis sample whose upstream `response_label` is "safe", drops the
rest, and refills from the remaining safe-labelled upstream rows (train + validation) until
the original token budget is matched. Refill rows are disjoint from the old sample, the
WildJailbreak and PKU-SafeRLHF samples, and the Aegis test split (exact normalized prompt
and response, and MinHash Jaccard >= 0.8 on 3-word shingles), and unique among themselves.

The unfiltered sample is first saved as `aegis_all.parquet`; the result is written to
`aegis.parquet`, the file the training step reads.

Run: python -m modal run data/build_aegis_safe_sample.py
"""
import hashlib
import json
import os
import random
import re
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (HF_CACHE_VOLUME, HF_SECRET_NAME, MAX_SEQ_LEN, VOL, VOL_BENCH,  # noqa: E402
                    VOL_EXT, VOLUME_NAME)

app = modal.App("bias-drift-aegis-safe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets", "pandas", "pyarrow", "transformers", "tokenizers", "jinja2",
                 "huggingface_hub", "hf_transfer", "datasketch")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("config", "dataset_registry")
)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME)

HF_ID = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
SAMPLE = f"{VOL_EXT}/sample/safety"
SEED = 4242
_WS, _P = re.compile(r"\s+"), re.compile(r"[^\w\s]")


def norm(s):
    return _WS.sub(" ", _P.sub(" ", str(s or "").lower())).strip()


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") >> 1


def usable(df):
    """Rows with a non-empty response, in upstream order (row_idx = position after filtering)."""
    df = df[df["response"].notna() & (df["response"].astype(str).str.strip() != "")]
    return df.reset_index(drop=True)


@app.function(image=image, volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)], timeout=3600, cpu=8, memory=16384)
def select():
    import pandas as pd
    from datasets import load_dataset
    from datasketch import MinHash, MinHashLSH
    from transformers import AutoTokenizer
    vol.reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

    def n_tok(p, r):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
        return min(len(tok(text, add_special_tokens=False)["input_ids"]), MAX_SEQ_LEN)

    # upstream rows; the sample's row_idx indexes the train split, validation rows are offset
    train = usable(load_dataset(HF_ID, split="train").to_pandas())
    train["row_idx"] = range(len(train))
    val = usable(load_dataset(HF_ID, split="validation").to_pandas())
    val["row_idx"] = range(10_000_000, 10_000_000 + len(val))
    up = pd.concat([train, val], ignore_index=True)
    label = up.set_index("row_idx")["response_label"]

    if not os.path.exists(f"{SAMPLE}/aegis_all.parquet"):
        os.replace(f"{SAMPLE}/aegis.parquet", f"{SAMPLE}/aegis_all.parquet")
    old = pd.read_parquet(f"{SAMPLE}/aegis_all.parquet")
    old["response_label"] = old["row_idx"].map(label)
    keep = old[old["response_label"] == "safe"]
    budget = int(old["n_tokens"].sum())
    print(f"[aegis] old n={len(old)} tok={budget:,}; keep safe={len(keep)} "
          f"({int(keep.n_tokens.sum()):,} tok)")

    # everything the refill must stay disjoint from
    claimed_p, claimed_r, texts = set(), set(), []
    for _, r in old.iterrows():
        claimed_p.add(h64(norm(r.prompt))); claimed_r.add(h64(norm(r.response))); texts.append(norm(r.prompt))
    for k in ("wildjailbreak", "pku_saferlhf", "pku_safe"):
        if os.path.exists(f"{SAMPLE}/{k}.parquet"):
            for p in pd.read_parquet(f"{SAMPLE}/{k}.parquet")["prompt"]:
                claimed_p.add(h64(norm(p))); texts.append(norm(p))
    for line in open(f"{VOL_BENCH}/aegis_test.jsonl", encoding="utf-8"):
        p = json.loads(line)["prompt"]
        claimed_p.add(h64(norm(p))); texts.append(norm(p))

    def minhash(text):
        m, w = MinHash(num_perm=128), text.split()
        for i in range(max(len(w) - 2, 1)):
            m.update(" ".join(w[i:i + 3]).encode())
        return m

    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    for i, t in enumerate(texts):
        lsh.insert(f"c{i}", minhash(t))

    # refill pool: safe-labelled rows not already in the sample, unique prompt and response
    pool = up[(up["response_label"] == "safe") & ~up["row_idx"].isin(set(old["row_idx"]))].copy()
    pool["_stratum"] = pool["violated_categories"].fillna("").astype(str)
    props = pool["_stratum"].value_counts(normalize=True).to_dict()
    pool["_ph"], pool["_rh"] = pool["prompt"].map(lambda p: h64(norm(p))), pool["response"].map(lambda p: h64(norm(p)))
    pool = pool.drop_duplicates("_ph").drop_duplicates("_rh")
    pool = pool[~pool["_ph"].isin(claimed_p) & ~pool["_rh"].isin(claimed_r)]
    by = {s: g.sample(frac=1, random_state=SEED).to_dict("records") for s, g in pool.groupby("_stratum")}

    counts = {s: 0 for s in by}
    target = budget - int(keep["n_tokens"].sum())
    added, add_tok = [], 0
    while add_tok < target - 64 and by:   # draw from the stratum most under-represented so far
        s = max(by, key=lambda s: props[s] * (1 + len(added)) - counts[s] if by[s] else -1e9)
        if not by[s]:
            by.pop(s); continue
        r = by[s].pop()
        t = n_tok(r["prompt"], r["response"])
        if add_tok + t > target:
            continue
        mh = minhash(norm(r["prompt"]))
        if lsh.query(mh):
            continue
        lsh.insert(f"a{len(added)}", mh)
        r["n_tokens"], r["stratum"] = t, s
        added.append(r); add_tok += t; counts[s] += 1
    shortfall = target - add_tok
    assert shortfall <= 0.15 * budget, f"safe pool exhausted (shortfall {shortfall:,} tok)"

    cols = ["stratum", "row_idx", "prompt", "response", "n_tokens"]
    out = pd.concat([keep[cols], pd.DataFrame(added)[cols]], ignore_index=True)
    assert (out["row_idx"].map(label) == "safe").all() and out["row_idx"].is_unique
    out.to_parquet(f"{SAMPLE}/aegis.parquet", index=False)
    report = {"old_rows": len(old), "kept_safe_rows": len(keep), "refilled_rows": len(added),
              "final_rows": len(out), "final_tokens": int(out["n_tokens"].sum()),
              "budget_tokens": budget, "token_shortfall": int(shortfall)}
    os.makedirs(f"{VOL_EXT}/dedup/safety", exist_ok=True)
    json.dump(report, open(f"{VOL_EXT}/dedup/safety/aegis_safe_report.json", "w"), indent=2)
    vol.commit()
    print("[aegis] DONE", json.dumps(report))
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(select.remote(), indent=2))
