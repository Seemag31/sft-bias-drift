"""PKU-SafeRLHF safe-only sample (the PKU cell reported in the paper).

Keeps the safe-labelled rows of the existing PKU sample, drops rows whose chosen
("safer") response is itself labelled unsafe, and refills from the safe-only pool,
stratified to the safe-subpopulation proportions and deduplicated against the other
two safety datasets, until the 1.0-million-token budget is matched.

Run: python -m modal run data/build_pku_safe_sample.py
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
from config import HF_CACHE_VOLUME, HF_SECRET_NAME, MAX_SEQ_LEN, TOKEN_BUDGET, VOL, VOL_EXT, VOLUME_NAME  # noqa: E402

app = modal.App("bias-drift-pku-safe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas", "pyarrow", "transformers", "tokenizers", "jinja2",
                 "huggingface_hub", "hf_transfer")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("config", "dataset_registry")
)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME)
EXT = VOL_EXT
BUDGET = TOKEN_BUDGET

_WS = re.compile(r"\s+")
_P = re.compile(r"[^\w\s]")


def norm(s):
    return _WS.sub(" ", _P.sub(" ", str(s or "").lower())).strip()


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") >> 1


@app.function(image=image, volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
              timeout=3600, cpu=8, memory=16384)
def select():
    import pandas as pd
    from transformers import AutoTokenizer
    vol.reload()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

    def n_tok(p, r):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False)
        return min(len(tok(text, add_special_tokens=False)["input_ids"]), MAX_SEQ_LEN)

    # 1. existing samples: keep pku's safe rows; block aegis/wjb claims
    old = pd.read_parquet(f"{EXT}/sample/safety/pku_saferlhf.parquet")
    keep = old[old.stratum.astype(str).str.startswith("sev0")].copy()
    dropped = old[~old.stratum.astype(str).str.startswith("sev0")]
    blocked = set()
    for k in ("aegis", "wildjailbreak"):
        s = pd.read_parquet(f"{EXT}/sample/safety/{k}.parquet")
        blocked |= {h64(norm(p)) for p in s["prompt"]}
    keep["_h"] = keep["prompt"].map(lambda p: h64(norm(p)))
    claimed = set(keep["_h"]) | blocked

    kept_tokens = int(keep["n_tokens"].sum())
    print(f"[pku_safe] kept {len(keep)} safe rows ({kept_tokens:,} tok); "
          f"dropped {len(dropped)} unsafe rows "
          f"({int(dropped['n_tokens'].sum()):,} tok to refill)")

    # 2. refill from the safe-only pool, stratified to full safe-census props
    pool = pd.read_parquet(f"{EXT}/pool/safety/pku_safe.parquet")
    meta = json.load(open(f"{EXT}/scan/safety/pku_safe.meta.json"))
    full = meta["strata_full"]; tot = sum(full.values())
    pool["_h"] = pool["prompt"].map(lambda p: h64(norm(p)))
    pool = pool.drop_duplicates("_h")
    pool = pool[~pool["_h"].isin(claimed)]
    rng = random.Random(4242)
    by_stratum = {s: g.sample(frac=1, random_state=4242).to_dict("records")
                  for s, g in pool.groupby("stratum")}
    # proportional interleave: repeatedly draw from the stratum most
    # under-represented relative to the safe census
    counts = {s: 0 for s in by_stratum}
    props = {s: full.get(s, 0) / tot for s in by_stratum}
    added, add_tok = [], 0
    target = BUDGET - kept_tokens
    while add_tok < target - 64:
        s = max(by_stratum, key=lambda s: props.get(s, 0) * (1 + len(added)) - counts[s]
                if by_stratum[s] else -1)
        if not by_stratum[s]:
            by_stratum.pop(s); counts.pop(s); props.pop(s, None)
            if not by_stratum:
                break
            continue
        r = by_stratum[s].pop()
        t = n_tok(r["prompt"], r["response"])
        if add_tok + t > target:
            continue
        r["n_tokens"] = t
        added.append(r); add_tok += t; counts[s] += 1
    print(f"[pku_safe] refilled {len(added)} rows (+{add_tok:,} tok)")

    out = pd.concat([keep.drop(columns=["_h"]),
                     pd.DataFrame(added)[["stratum", "row_idx", "prompt",
                                          "response", "n_tokens"]]],
                    ignore_index=True)
    total = int(out["n_tokens"].sum())
    out.to_parquet(f"{EXT}/sample/safety/pku_safe.parquet", index=False)
    rep = {"kept_safe_rows": int(len(keep)), "dropped_unsafe_rows": int(len(dropped)),
           "refilled_rows": int(len(added)), "final_n": int(len(out)),
           "final_tokens": total, "budget": BUDGET,
           "safe_pool_unique": int(len(pool))}
    with open(f"{EXT}/dedup/safety/pku_safe_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    vol.commit()
    print(f"[pku_safe] DONE n={len(out)} tokens={total:,}")
    return rep


@app.local_entrypoint()
def main():
    print(json.dumps(select.remote(), indent=2))
