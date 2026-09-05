"""Stage 0 of data preparation: put BBQ and the capability benchmarks on the volume.

  data/bbq/<category>.jsonl      official BBQ (nyu-mll/BBQ), 9 categories
  data/benchmarks/gsm8k_test.jsonl, humaneval.jsonl, medqa_test.jsonl, aegis_test.jsonl

Run: python -m modal run data/download_benchmarks.py
"""
import json
import os
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (  # noqa: E402
    BBQ_CATEGORIES, BBQ_RAW_URL, HF_CACHE_VOLUME, HF_SECRET_NAME, VOL, VOL_BBQ,
    VOL_BENCH, VOLUME_NAME,
)

app = modal.App("bias-drift-benchmarks")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("datasets==4.4.1", "huggingface_hub", "hf_transfer", "requests")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
         .add_local_python_source("config"))
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(image=image, volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)], timeout=3600)
def download():
    import requests
    from datasets import load_dataset

    os.makedirs(VOL_BBQ, exist_ok=True)
    os.makedirs(VOL_BENCH, exist_ok=True)
    counts = {}
    for cat in BBQ_CATEGORIES:
        path = f"{VOL_BBQ}/{cat}.jsonl"
        if not os.path.exists(path):
            r = requests.get(BBQ_RAW_URL.format(cat=cat), timeout=120)
            r.raise_for_status()
            open(path, "w", encoding="utf-8").write(r.text)
        counts[f"bbq/{cat}"] = sum(1 for _ in open(path, encoding="utf-8"))

    def write(name, rows):
        path = f"{VOL_BENCH}/{name}.jsonl"
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = sum(1 for _ in open(path, encoding="utf-8"))

    write("gsm8k_test", ({"question": r["question"], "answer": r["answer"]}
                         for r in load_dataset("openai/gsm8k", "main", split="test")))
    write("humaneval", (dict(r) for r in load_dataset("openai/openai_humaneval", split="test")))
    write("medqa_test", (dict(r) for r in load_dataset("GBaker/MedQA-USMLE-4-options", split="test")))
    write("aegis_test", ({"prompt": r["prompt"], "label": r["prompt_label"]}
                         for r in load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="test")
                         if (r.get("prompt") or "").strip() and r.get("prompt_label") in ("safe", "unsafe")))
    vol.commit()
    print(json.dumps(counts, indent=2))
    return counts


@app.local_entrypoint()
def main():
    download.remote()
