"""Stage 1 of data preparation: one streaming pass over every dataset (CPU-only Modal job).

For every row this records, without keeping the full text:
  * stratum    sub-category used for proportional sampling
  * h64        hash of the normalised prompt (exact-match deduplication)
  * bbq_*      BBQ contamination checks (template patterns, exact questions,
               verbatim context reuse)
and writes a stratified candidate pool (4x the nominal target) with text so the
later stages never re-download.

Outputs on the volume: ext/scan/<domain>/<key>.parquet, ext/pool/<domain>/<key>.parquet,
ext/scan/<domain>/<key>.meta.json

Run one:   python -m modal run data/scan_datasets.py --domain math --key openr1
Run all:   python -m modal run data/scan_datasets.py
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
from config import HF_CACHE_VOLUME, HF_SECRET_NAME, VOLUME_NAME, VOL, VOL_EXT  # noqa: E402
from dataset_registry import EXT_DATASETS, TARGET_N  # noqa: E402

app = modal.App("bias-drift-scan")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets==4.4.1", "huggingface_hub", "hf_transfer", "requests",
                 "pandas", "pyarrow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
    .add_local_python_source("config", "dataset_registry")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME)

POOL_MULT = 4          # candidate pool = 4x target, so dedup has refill headroom
RESERVOIR = 120_000    # rows kept with text during the scan (uniform reservoir)
TEXT_CAP = 20_000      # chars of prompt/response retained in the pool
SHINGLE_K = 8          # word-shingle length for BBQ context reuse detection
EXT = VOL_EXT

# ------------------------------------------------------------------ BBQ grep
# Ordered; index i sets bit i in bbq_tmpl.
BBQ_PATTERNS = [
    ("who_more_likely", r"\bwho (?:was|is|were|are|would be) (?:more |most )?likely\b"),
    ("which_more_likely", r"\bwhich (?:one |person |of them )?(?:was|is) (?:more|most) likely\b"),
    ("cant_determine_option", r"\b(?:can'?t|cannot|could not) be determined\b"),
    ("not_enough_info", r"\bnot enough (?:info|information)\b"),
    ("undetermined_option", r"\b(?:undetermined|not answerable|unknown answer)\b"),
    ("cant_answer", r"\b(?:can'?t|cannot) answer\b"),
    ("bbq_named", r"\bBBQ\b|\bBias Benchmark for (?:QA|Question Answering)\b"),
    ("bbq_schema_fields", r"\b(?:context_condition|stereotyped_groups|question_polarity|"
                          r"answer_info|disambig(?:uated)?_context|ambig(?:uous)?_context)\b"),
    ("ambig_disambig", r"\bambig\b|\bdisambig\b"),
    ("stereotype_qa", r"\bstereotyp\w*\b.{0,80}\b(?:question|answer|option|benchmark|bias)\b"),
    ("neg_qa_frame", r"\bwho (?:was|is) (?:the )?(?:more |most )?"
                     r"(?:violent|dangerous|aggressive|lazy|incompetent|poor|rich|smart)\b"),
]
BBQ_RX = [(n, re.compile(p, re.I)) for n, p in BBQ_PATTERNS]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm_text(s: str) -> str:
    s = _PUNCT.sub(" ", str(s or "").lower())
    return _WS.sub(" ", s).strip()


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "big", signed=False) >> 1   # keep it positive/int64-safe


def pku_extract(row):
    """PKU-SafeRLHF is a preference dataset; the SFT view is prompt -> the
    SAFER of the two responses. Stratum encodes severity + primary harm
    category of that chosen response so sampling preserves the harm mix."""
    sid = row.get("safer_response_id")
    if sid not in (0, 1):
        return None
    resp = row.get(f"response_{sid}") or ""
    safe = row.get(f"is_response_{sid}_safe")
    sev = row.get(f"response_{sid}_severity_level")
    harm = row.get(f"response_{sid}_harm_category") or {}
    cats = sorted(k for k, v in harm.items() if v)
    stratum = f"sev{sev}|{cats[0] if cats else ('none' if safe else 'unlabeled')}"
    return {"prompt": row.get("prompt") or "", "response": resp,
            "_stratum": stratum}


def get_field(row, col):
    """Supports dotted access for nested metadata columns (e.g. metadata.extension)."""
    if col is None:
        return None
    cur = row
    for part in col.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def row_prompt(row, spec):
    """The 'question' text used for dedup + BBQ matching."""
    if spec.get("prompt_col"):
        p = row.get(spec["prompt_col"])
        if spec.get("input_col") and row.get(spec["input_col"]):
            p = f"{p}\n{row[spec['input_col']]}"
        return p or ""
    msgs = row.get(spec.get("messages_col") or "")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content") or ""
    return ""


def row_response(row, spec):
    if spec.get("response_col"):
        return row.get(spec["response_col"]) or ""
    msgs = row.get(spec.get("messages_col") or "")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content") or ""
    return ""


@app.function(image=image, volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
              timeout=6 * 3600, cpu=8, memory=32768)
def scan_one(domain: str, key: str, force: bool = False):
    import pandas as pd
    from datasets import load_dataset

    spec = next(s for s in EXT_DATASETS[domain] if s["key"] == key)
    scan_dir, pool_dir = f"{EXT}/scan/{domain}", f"{EXT}/pool/{domain}"
    os.makedirs(scan_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)
    scan_p = f"{scan_dir}/{key}.parquet"
    if os.path.exists(scan_p) and not force:
        print(f"[{domain}/{key}] scan exists, skipping")
        return json.load(open(f"{scan_dir}/{key}.meta.json"))

    # ---- real BBQ questions/context-shingles, for exact contamination matching
    bbq_qs, bbq_shingles = load_bbq_reference()
    print(f"[{domain}/{key}] BBQ reference: {len(bbq_qs)} question hashes, "
          f"{len(bbq_shingles)} context shingle hashes")

    print(f"[{domain}/{key}] streaming {spec['hf_id']} "
          f"cfg={spec['config']} split={spec['split']} "
          f"data_files={spec.get('data_files')}")
    if spec.get("data_files"):
        # subset selection by repo path (e.g. UltraData English-Math no_think)
        ds = load_dataset("json", split="train", streaming=True,
                          data_files=f"hf://datasets/{spec['hf_id']}/{spec['data_files']}")
    else:
        ds = load_dataset(spec["hf_id"], spec["config"], split=spec["split"],
                          streaming=True)
    row_filter = spec.get("row_filter")
    n_filtered = 0

    recs = {"stratum": [], "h64": [], "bbq_tmpl": [], "bbq_exact": [],
            "bbq_ctx": [], "n_chars_p": [], "n_chars_r": []}
    pool_keep, hit_examples = [], []
    rng = random.Random(1234)
    strat_col = spec.get("strat_col")
    n = 0
    for row in ds:
        if row_filter is not None:
            v = row.get(row_filter["col"])
            bad = (not str(v or "").strip()) if row_filter.get("nonempty") \
                else v != row_filter.get("equals")
            if bad:
                n_filtered += 1
                continue
        if spec.get("special") in ("pku_safer_response", "pku_safer_response_safe_only"):
            safe_only = spec.get("special").endswith("safe_only")
            row = pku_extract(row)
            if row is None or (safe_only and not str(row["_stratum"]).startswith("sev0")):
                n_filtered += 1
                continue
        elif spec.get("special") == "wjb_prompt":
            if not (row.get("adversarial") or "").strip():
                row = dict(row, adversarial=row.get("vanilla") or "")
        p, r = row_prompt(row, spec), row_response(row, spec)
        np_ = norm_text(p)
        mask = 0
        for i, (_, rx) in enumerate(BBQ_RX):
            if rx.search(p) or rx.search(r[:4000]):
                mask |= 1 << i
        hp = h64(np_)
        exact = hp in bbq_qs
        # near-verbatim BBQ context reuse: does this row contain >=2 distinct
        # 8-word shingles that also occur in an official BBQ context?
        ctxhit = False
        if len(np_) > 60:
            toks = np_.split()
            if len(toks) >= SHINGLE_K:
                seen = 0
                for i in range(0, len(toks) - SHINGLE_K + 1):
                    if h64(" ".join(toks[i:i + SHINGLE_K])) in bbq_shingles:
                        seen += 1
                        if seen >= 2:
                            ctxhit = True
                            break
        st = row.get("_stratum") or (get_field(row, strat_col) if strat_col else "__all__")
        st = "__none__" if st is None else str(st)
        recs["stratum"].append(st)
        recs["h64"].append(hp)
        recs["bbq_tmpl"].append(mask)
        recs["bbq_exact"].append(exact)
        recs["bbq_ctx"].append(ctxhit)
        recs["n_chars_p"].append(len(p))
        recs["n_chars_r"].append(len(r))
        if (mask or exact or ctxhit) and len(hit_examples) < 200:
            hit_examples.append({
                "row": n, "mask": mask, "exact": bool(exact), "ctx": bool(ctxhit),
                "patterns": [BBQ_PATTERNS[i][0] for i in range(len(BBQ_RX))
                             if mask >> i & 1],
                "prompt": p[:400],
            })
        # Reservoir sample of rows WITH text, so later stages (MinHash, tokenizer,
        # final selection) never re-download. Uniform over the whole stream, then
        # re-stratified to the full-dataset proportions once the scan finishes.
        # cap stored text: MAX_SEQ_LEN is 2048 tokens, so ~20k chars is already
        # far past anything training will see, and it bounds reservoir memory
        # (starcoder responses are whole source files).
        item = (n, st, p[:TEXT_CAP], r[:TEXT_CAP])
        if len(pool_keep) < RESERVOIR:
            pool_keep.append(item)
        else:
            j = rng.randrange(n + 1)
            if j < RESERVOIR:
                pool_keep[j] = item
        n += 1
        if n % 200_000 == 0:
            print(f"  .. {n:,} rows", flush=True)

    scan = pd.DataFrame(recs)
    scan["row_idx"] = range(len(scan))
    scan.to_parquet(scan_p, index=False)

    # ---- stratified candidate pool, proportional to the FULL dataset
    pool_df = pd.DataFrame(pool_keep, columns=["row_idx", "stratum", "prompt", "response"])
    want = min(POOL_MULT * TARGET_N, len(pool_df))
    pool = stratified_take(pool_df, scan["stratum"].value_counts(normalize=True),
                           want, seed=1234)
    pool.to_parquet(f"{pool_dir}/{key}.parquet", index=False)

    dist = scan["stratum"].value_counts()
    meta = {
        "domain": domain, "key": key, "hf_id": spec["hf_id"],
        "config": spec["config"], "split": spec["split"],
        "n_rows": int(n),
        "n_rows_filtered_out": int(n_filtered),
        "row_filter": row_filter,
        "strat_col": strat_col,
        "n_strata": int(dist.size),
        "strata_full": {k: int(v) for k, v in dist.head(200).items()},
        "pool_rows": int(len(pool)),
        "pool_strata": {k: int(v) for k, v in pool["stratum"].value_counts().head(200).items()},
        "bbq_template_hits": int((scan["bbq_tmpl"] > 0).sum()),
        "bbq_template_by_pattern": {
            BBQ_PATTERNS[i][0]: int(((scan["bbq_tmpl"].to_numpy() >> i) & 1).sum())
            for i in range(len(BBQ_RX))},
        "bbq_exact_question_hits": int(scan["bbq_exact"].sum()),
        "bbq_context_substring_hits": int(scan["bbq_ctx"].sum()),
        "n_unique_questions": int(scan["h64"].nunique()),
        "within_dataset_dup_rate": float(1 - scan["h64"].nunique() / max(n, 1)),
        "mean_chars_prompt": float(scan["n_chars_p"].mean()),
        "mean_chars_response": float(scan["n_chars_r"].mean()),
        "hit_examples": hit_examples[:60],
    }
    with open(f"{scan_dir}/{key}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()
    print(f"[{domain}/{key}] DONE n={n:,} strata={dist.size} "
          f"bbq_tmpl={meta['bbq_template_hits']:,} "
          f"bbq_exact={meta['bbq_exact_question_hits']} "
          f"bbq_ctx={meta['bbq_context_substring_hits']}")
    return meta


def stratified_take(df, target_props, n_want, seed):
    """Largest-remainder proportional allocation across strata, seeded sampling."""
    import pandas as pd
    rng = random.Random(seed)
    groups = {s: list(ix) for s, ix in df.groupby("stratum").groups.items()}
    exact = {s: n_want * float(target_props.get(s, 0.0)) for s in groups}
    quota = {s: int(v) for s, v in exact.items()}
    rem = n_want - sum(quota.values())
    for s in sorted(exact, key=lambda s: exact[s] - quota[s], reverse=True)[:max(rem, 0)]:
        quota[s] += 1
    picks = []
    for s in sorted(groups):
        k = min(quota.get(s, 0), len(groups[s]))
        picks += rng.sample(groups[s], k)
    return df.loc[sorted(picks)].reset_index(drop=True)


def load_bbq_reference():
    """(normalized-question hashes, 8-word context-shingle hashes) from official BBQ."""
    import requests
    from config import BBQ_CATEGORIES, BBQ_RAW_URL, VOL_BBQ
    qs, shingles = set(), set()
    os.makedirs(VOL_BBQ, exist_ok=True)
    for cat in BBQ_CATEGORIES:
        path = f"{VOL_BBQ}/{cat}.jsonl"
        if not os.path.exists(path):
            r = requests.get(BBQ_RAW_URL.format(cat=cat), timeout=120)
            r.raise_for_status()
            open(path, "w", encoding="utf-8").write(r.text)
        for line in open(path, encoding="utf-8"):
            ex = json.loads(line)
            qs.add(h64(norm_text(ex["question"])))
            toks = norm_text(ex["context"]).split()
            for i in range(0, max(len(toks) - SHINGLE_K + 1, 0)):
                shingles.add(h64(" ".join(toks[i:i + SHINGLE_K])))
    return qs, shingles


@app.local_entrypoint()
def main(domain: str = "", key: str = "", force: bool = False, skip_gated: bool = True):
    if domain and key:
        jobs = [(domain, key, force)]
    else:
        jobs = [(d, s["key"], force) for d, specs in EXT_DATASETS.items()
                for s in specs if not (skip_gated and s.get("gated"))]
        skipped = [s["hf_id"] for _, specs in EXT_DATASETS.items()
                   for s in specs if s.get("gated")]
        if skip_gated and skipped:
            print("SKIPPING gated (no access with the current HF token): "
                  + ", ".join(skipped))
    print(f"launching {len(jobs)} scans in parallel")
    res = []
    for out in scan_one.starmap(jobs, return_exceptions=True):
        res.append(out if not isinstance(out, Exception)
                   else {"error": f"{type(out).__name__}: {out}"[:500]})
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "logs", "scan_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    for m in res:
        if "error" in m:
            print("ERROR:", m["error"][:200])
        else:
            print(f"{m['domain']}/{m['key']:<14} n={m['n_rows']:>10,} "
                  f"strata={m['n_strata']:>5} "
                  f"bbq_tmpl={m['bbq_template_hits']:>7,} "
                  f"bbq_exact={m['bbq_exact_question_hits']:>4} "
                  f"bbq_ctx={m['bbq_context_substring_hits']:>4} "
                  f"dupes={m['within_dataset_dup_rate']:.3f}")
