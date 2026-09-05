"""Stage 2 of data preparation: deduplicate and draw the token-matched training samples.

Per domain (one CPU-only Modal call each):
  1. within-dataset exact deduplication (normalised prompt hash, keep first);
  2. datasets in priority order (anchor first): fill a proportional stratified
     sample, rejecting any prompt that exact- or near-matches (MinHash/LSH,
     Jaccard >= 0.8, 3-word shingles) one already claimed by an earlier dataset;
  3. stop when the 1.0-million-token budget is reached (Qwen3.5-9B tokenizer,
     chat-formatted, capped at MAX_SEQ_LEN), so every run trains on the same
     number of tokens while sample sizes differ.

Outputs: ext/sample/<domain>/<key>.parquet and ext/dedup/<domain>/report.json

Run:  python -m modal run --detach data/build_samples.py             # all domains
      python -m modal run data/build_samples.py --domain medical      # one domain
"""
import json
import os
import random
import re
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (  # noqa: E402
    HF_CACHE_VOLUME, HF_SECRET_NAME, MAX_SEQ_LEN, TOKEN_BUDGET, VOL, VOL_EXT, VOLUME_NAME,
)
from dataset_registry import EXT_DATASETS, TARGET_N  # noqa: E402

app = modal.App("bias-drift-samples")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas", "pyarrow", "datasketch==1.6.5", "transformers",
                 "huggingface_hub", "hf_transfer", "tokenizers", "jinja2")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("config", "dataset_registry")
)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME)

EXT = VOL_EXT
GLOBAL_BUDGET = TOKEN_BUDGET   # one budget for every dataset in every domain
JACCARD = 0.8
NUM_PERM = 128
SHINGLE_W = 3          # word 3-shingles for MinHash
REF_TOKENIZER = "Qwen/Qwen3.5-9B"   # single reference tokenizer for budgets

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm_text(s: str) -> str:
    s = _PUNCT.sub(" ", str(s or "").lower())
    return _WS.sub(" ", s).strip()


def h64(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "big") >> 1


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(r: str) -> str:
    """Remove inline reasoning traces. If a trace never closes (truncated in
    the candidate pool), the post-trace answer is unrecoverable -> empty."""
    r = str(r or "")
    if "</think>" in r:
        r = r.rsplit("</think>", 1)[1] if r.lstrip().startswith("<think>") \
            else _THINK.sub("", r)
    elif r.lstrip().startswith("<think>"):
        return ""
    return r.strip()


def minhash(norm: str):
    from datasketch import MinHash
    m = MinHash(num_perm=NUM_PERM)
    toks = norm.split()
    if len(toks) < SHINGLE_W:
        m.update(norm.encode())
        return m
    for i in range(len(toks) - SHINGLE_W + 1):
        m.update(" ".join(toks[i:i + SHINGLE_W]).encode())
    return m


@app.function(image=image, volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
              timeout=4 * 3600, cpu=8, memory=32768,
              max_containers=1)
def process_domain(domain: str, force: bool = False):
    import pandas as pd
    from datasketch import MinHashLSH
    from transformers import AutoTokenizer

    out_dir = f"{EXT}/sample/{domain}"
    rep_p = f"{EXT}/dedup/{domain}/report.json"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(rep_p), exist_ok=True)
    if os.path.exists(rep_p) and not force:
        print(f"[{domain}] already processed, skipping")
        return json.load(open(rep_p))

    specs = sorted(EXT_DATASETS[domain], key=lambda s: s["priority"])
    specs = [s for s in specs
             if os.path.exists(f"{EXT}/pool/{domain}/{s['key']}.parquet")]
    print(f"[{domain}] datasets present: {[s['key'] for s in specs]}")

    tok = AutoTokenizer.from_pretrained(REF_TOKENIZER)

    def n_tokens(p, r):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert isinstance(ids, list) and (not ids or isinstance(ids[0], int))
        return min(len(ids), MAX_SEQ_LEN)

    lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    claimed_h64 = set()
    report = {"domain": domain, "target_n": TARGET_N, "jaccard": JACCARD,
              "datasets": {}}
    selections = {}

    for spec in specs:
        key = spec["key"]
        pool = pd.read_parquet(f"{EXT}/pool/{domain}/{key}.parquet")
        # full strata census from the per-row scan parquet
        scan_strata = pd.read_parquet(
            f"{EXT}/scan/{domain}/{key}.parquet", columns=["stratum"]
        )["stratum"].value_counts()
        full_strata = {str(k): int(v) for k, v in scan_strata.items()}
        total_full = sum(full_strata.values())

        # strip inline <think> traces (X-Coder) to keep the response-only format
        if key == "xcoder":
            pool["response"] = pool["response"].map(strip_think)
            n_empty = int((pool["response"].str.strip() == "").sum())
            pool = pool[pool["response"].str.strip() != ""]
            print(f"[{domain}/{key}] stripped <think>; dropped {n_empty} rows "
                  f"left empty (unclosed trace within TEXT_CAP)")

        # ---- within-dataset dedup (keep first)
        pool["_norm"] = pool["prompt"].map(norm_text)
        pool["_h"] = pool["_norm"].map(h64)
        n0 = len(pool)
        pool = pool.drop_duplicates("_h", keep="first")
        within_dropped = n0 - len(pool)

        # ---- proportional interleaved fill straight to the GLOBAL token budget.
        # Every candidate gets key = (rank_within_stratum + 0.5) / stratum_share;
        # walking keys in ascending order keeps every prefix of the selection
        # proportional to the full dataset, so "stop when the budget is full"
        # and "skip duplicates, the stratum's next candidate refills" both
        # preserve stratification by construction.
        props = {s: c / total_full for s, c in full_strata.items()}
        order = []
        for s, g in pool.groupby("stratum"):
            share = props.get(str(s), 0.0)
            if share <= 0:
                continue
            idxs = list(g.index)
            random.Random(4242).shuffle(idxs)
            order += [((r + 0.5) / share, idx, s) for r, idx in enumerate(idxs)]
        order.sort()

        picked, exact_drops, near_drops, within_near_drops = [], 0, 0, 0
        tot = 0
        for _, idx, s in order:
            if GLOBAL_BUDGET - tot < 64:
                break
            row = pool.loc[idx]
            if row["_h"] in claimed_h64:
                exact_drops += 1
                continue
            mh = minhash(row["_norm"])
            hits = lsh.query(mh)
            if hits:
                if any(h.split(":", 1)[0] != key for h in hits):
                    near_drops += 1
                else:
                    within_near_drops += 1
                continue
            t = n_tokens(row["prompt"], row["response"])
            if tot + t > GLOBAL_BUDGET:
                continue    # keep walking; a shorter later candidate may fit
            picked.append((s, row, t))
            claimed_h64.add(row["_h"])
            lsh.insert(f"{key}:{row['row_idx']}", mh)
            tot += t
        sel = pd.DataFrame(
            [{"stratum": s, "row_idx": r["row_idx"], "prompt": r["prompt"],
              "response": r["response"], "h64": r["_h"], "n_tokens": t}
             for s, r, t in picked])
        budget_shortfall = max(GLOBAL_BUDGET - tot, 0) if tot < GLOBAL_BUDGET * 0.995 else 0
        selections[key] = sel
        sel.to_parquet(f"{out_dir}/{key}.parquet", index=False)
        report["datasets"][key] = {
            "hf_id": spec["hf_id"], "priority": spec["priority"],
            "pool_rows": int(n0), "within_dataset_dups_dropped": int(within_dropped),
            "cross_dataset_exact_drops": int(exact_drops),
            "cross_dataset_near_drops_minhash": int(near_drops),
            "within_dataset_near_drops_minhash": int(within_near_drops),
            "final_n": int(len(sel)),
            "final_tokens": int(sel["n_tokens"].sum()),
            "mean_tokens": float(sel["n_tokens"].mean()),
            "budget_shortfall_tokens": int(budget_shortfall),
            "n_strata_represented": int(sel["stratum"].nunique()),
            "final_strata_head": {str(k): int(v) for k, v in
                                  sel["stratum"].value_counts().head(30).items()},
        }
        print(f"[{domain}/{key}] n={len(sel)} tokens={sel['n_tokens'].sum():,} "
              f"exact_drops={exact_drops} near_drops={near_drops} "
              f"budget_shortfall={budget_shortfall}")
    report["token_budget"] = GLOBAL_BUDGET

    # ---------------- verification: zero duplicates across final samples
    all_h = []
    for key in selections:
        fin = pd.read_parquet(f"{out_dir}/{key}.parquet")
        all_h += [(key, h) for h in fin["h64"]]
    seen, cross_dupes = {}, 0
    for key, h in all_h:
        if h in seen and seen[h] != key:
            cross_dupes += 1
        seen[h] = key
    report["final_cross_dataset_exact_dupes"] = int(cross_dupes)

    with open(rep_p, "w") as f:
        json.dump(report, f, indent=2)
    vol.commit()
    print(f"[{domain}] DONE budget={GLOBAL_BUDGET:,} cross_dupes_remaining={cross_dupes}")
    return report


@app.local_entrypoint()
def main(domain: str = "", force: bool = False):
    domains = [domain] if domain else list(EXT_DATASETS)
    res = []
    for out in process_domain.starmap([(d, force) for d in domains],
                                      return_exceptions=True):
        res.append(out if not isinstance(out, Exception)
                   else {"error": f"{type(out).__name__}: {out}"[:500]})
    with open(os.path.join(ROOT, "results", "logs", "sample_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    for r in res:
        if "error" in r:
            print("ERROR:", r["error"])
        else:
            print(f"{r['domain']}: budget={r.get('token_budget'):,} "
                  f"cross_dupes_remaining={r.get('final_cross_dataset_exact_dupes')}")
