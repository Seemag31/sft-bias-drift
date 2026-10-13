"""LoRA fine-tuning on Modal (detached, serial: one GPU container at a time).

Data preparation (CPU) writes one train.jsonl per run under ext/final/:
  <domain>_normal   the anchor sample as-is                       (standard SFT)
  <domain>_idk      ~10% of tokens replaced by CoCoNot abstention examples
                    (Experiment 2, abstention-augmented SFT)
  <domain>_<key>    each additional dataset sample                (main grid)

Training: bf16 LoRA on all linear projections, r=16, alpha=32, dropout 0.05,
lr 2e-4 with cosine decay and 3% warmup, effective batch 16, one epoch, prompt
tokens masked from the loss, checkpoint/resume on the volume, adapter merged for
vLLM. Every run records run_config.json (hyperparameters, loss, hours, cost).

Entrypoints:
  python -m modal deploy training/train_lora.py
  python -m modal run --detach training/train_lora.py::launch_all       # anchors: s1, s2, idk
  python -m modal run --detach training/train_lora.py::launch_additions # the 8 additional datasets
  python -m modal run --detach training/train_lora.py --run-name ext_math_s1 --model-key qwen
"""
import json
import os
import sys
import time

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (  # noqa: E402
    CHAT_TEMPLATE_KWARGS, GPU_PRICE_PER_HR, GPU_TYPE, GRAD_ACCUM,
    HF_CACHE_VOLUME, HF_SECRET_NAME, LEARNING_RATE, LOG_STEPS, LORA_ALPHA,
    LORA_DROPOUT, LORA_R, MAX_SEQ_LEN, MODELS, NUM_EPOCHS, PER_DEVICE_BATCH,
    SAVE_STEPS, VOL, VOL_RUNS, VOLUME_NAME,
)
from config import (  # noqa: E402
    ANCHORS, RUNS, IDK_CATEGORIES, IDK_FRAC, IDK_SOURCE, MODEL_ORDER,
    RUN_ORDER, RUN_ORDER_ADDITIONS, ADDITIONS, VOL_EXT, VOL_FINAL,
)

app = modal.App("bias-drift-train")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers", "peft", "accelerate", "datasets",
        "sentencepiece", "protobuf", "hf_transfer", "huggingface_hub",
        "pillow", "torchvision", "jinja2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("config", "dataset_registry")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets==4.4.1", "pandas", "pyarrow", "transformers",
                 "tokenizers", "jinja2", "huggingface_hub", "hf_transfer")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_python_source("config", "dataset_registry")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME)

EXT, FINAL = VOL_EXT, VOL_FINAL
REF_TOKENIZER = "Qwen/Qwen3.5-9B"

PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXCLUDE_TOWERS = ("vision", "visual", "audio", "image", "multi_modal", "multimodal")


# ======================================================================
# data preparation: standard + abstention-augmented train files per domain
# ======================================================================
@app.function(image=cpu_image, volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
              timeout=3600, cpu=8, memory=16384)
def prepare_final(force: bool = False):
    import pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(FINAL, exist_ok=True)
    marker = f"{FINAL}/_PREPARED"
    if os.path.exists(marker) and not force:
        print("[prepare] already done")
        return json.load(open(f"{FINAL}/prepare_meta.json"))

    tok = AutoTokenizer.from_pretrained(REF_TOKENIZER)

    def n_tok(p, r):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False)
        return min(len(tok(text, add_special_tokens=False)["input_ids"]),
                   MAX_SEQ_LEN)

    # ---- IDK pool from coconot
    cc = load_dataset(IDK_SOURCE, "original", split="train")
    cats = sorted(set(cc["category"]))
    keep_cats = [c for c in cats
                 if any(k in c.lower() for k in IDK_CATEGORIES)]
    print(f"[prepare] coconot categories: {cats} -> using {keep_cats}")
    idk_rows = [{"prompt": r["prompt"], "response": r["response"],
                 "stratum": f"__idk__:{r['category']}"}
                for r in cc if r["category"] in keep_cats]
    import random as _rnd
    _rnd.Random(4242).shuffle(idk_rows)
    for r in idk_rows:
        r["n_tokens"] = n_tok(r["prompt"], r["response"])

    meta = {"idk_source": IDK_SOURCE, "idk_categories_used": keep_cats,
            "idk_pool": len(idk_rows), "domains": {}}

    for domain, anchor in ANCHORS.items():
        sel = pd.read_parquet(f"{EXT}/sample/{domain}/{anchor}.parquet")
        total = int(sel["n_tokens"].sum())

        def write(rows, name):
            d = f"{FINAL}/{name}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/train.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({"prompt": r["prompt"],
                                        "response": r["response"],
                                        "stratum": str(r["stratum"])},
                                       ensure_ascii=False) + "\n")
            return len(rows)

        n_norm = write(sel.to_dict("records"), f"{domain}_normal")

        # ---- IDK arm: REPLACE ~10% of tokens, preserving stratification.
        # Removal walks the proportional interleave from the END (mirror of
        # selection order), so what remains keeps full-dataset proportions.
        target_remove = int(total * IDK_FRAC)
        order = []
        for s, g in sel.groupby("stratum"):
            for rank, idx in enumerate(g.index):
                order.append(((rank + 0.5) / len(g), idx))
        order.sort(reverse=True)
        removed_tok, removed_idx = 0, set()
        for _, idx in order:
            if removed_tok >= target_remove:
                break
            removed_idx.add(idx)
            removed_tok += int(sel.at[idx, "n_tokens"])
        kept = sel.loc[[i for i in sel.index if i not in removed_idx]]

        add, add_tok = [], 0
        for r in idk_rows:
            if removed_tok - add_tok < 32:
                break
            if add_tok + r["n_tokens"] > removed_tok:
                continue
            add.append(r)
            add_tok += r["n_tokens"]
        idk_all = kept.to_dict("records") + add
        _rnd.Random(4242).shuffle(idk_all)
        n_idk = write(idk_all, f"{domain}_idk")

        meta["domains"][domain] = {
            "anchor": anchor, "normal_n": n_norm, "normal_tokens": total,
            "idk_n": n_idk, "idk_removed_examples": len(removed_idx),
            "idk_removed_tokens": removed_tok, "idk_added_examples": len(add),
            "idk_added_tokens": add_tok,
            "idk_total_tokens": int(kept["n_tokens"].sum()) + add_tok,
            "idk_share_of_tokens": add_tok / max(int(kept["n_tokens"].sum()) + add_tok, 1),
        }
        print(f"[prepare] {domain}: normal n={n_norm} ({total:,} tok) | "
              f"idk n={n_idk} (-{len(removed_idx)} ex/-{removed_tok:,} tok, "
              f"+{len(add)} idk ex/+{add_tok:,} tok)")

    with open(f"{FINAL}/prepare_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    open(marker, "w").close()
    vol.commit()
    return meta


@app.function(image=cpu_image, volumes={VOL: vol},
              timeout=1800, cpu=4, memory=8192)
def prepare_additions(force: bool = False):
    """Write final/<domain>_<key>/train.jsonl for every additional dataset from
    its token-matched sample parquet."""
    import pandas as pd
    vol.reload()
    marker = f"{FINAL}/_PREPARED_ADDITIONS"
    if os.path.exists(marker) and not force:
        print("[prepare-add] already done")
        return "already_done"
    meta = {}
    for domain, keys in ADDITIONS.items():
        for key in keys:
            p = f"{EXT}/sample/{domain}/{key}.parquet"
            if not os.path.exists(p):
                print(f"[prepare-add] WARNING {domain}/{key} sample missing — skipped")
                meta[f"{domain}_{key}"] = {"skipped": "sample missing"}
                continue
            sel = pd.read_parquet(p)
            d = f"{FINAL}/{domain}_{key}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/train.jsonl", "w", encoding="utf-8") as f:
                for r in sel.to_dict("records"):
                    f.write(json.dumps({"prompt": r["prompt"],
                                        "response": r["response"],
                                        "stratum": str(r["stratum"])},
                                       ensure_ascii=False) + "\n")
            meta[f"{domain}_{key}"] = {"n": len(sel),
                                       "tokens": int(sel["n_tokens"].sum())}
            print(f"[prepare-add] {domain}/{key}: n={len(sel)} "
                  f"tokens={int(sel['n_tokens'].sum()):,}")
    with open(f"{FINAL}/prepare_additions_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    open(marker, "w").close()
    vol.commit()
    return meta


# ======================================================================
# training (per-run seed, serial)
# ======================================================================
def find_lora_targets(model):
    import torch.nn as nn
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(t in name.lower() for t in EXCLUDE_TOWERS):
            continue
        if name.split(".")[-1] in PROJ_NAMES:
            targets.append(name)
    if not targets:
        raise RuntimeError("No LoRA target modules found")
    return targets


@app.function(image=train_image, gpu=GPU_TYPE,
              volumes={VOL: vol, "/hf-cache": hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
              timeout=12 * 3600, memory=65536, cpu=8,
              retries=modal.Retries(max_retries=3, initial_delay=30.0),
              max_containers=1)
def train(run_name: str, model_key: str):
    import torch
    import transformers
    from transformers import (AutoConfig, AutoTokenizer, Trainer,
                              TrainingArguments, TrainerCallback,
                              DataCollatorForSeq2Seq, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset

    # Warm containers keep a stale volume snapshot; reload so this call sees
    # current data files and done-markers.
    vol.reload()
    spec = RUNS[run_name]
    assert model_key in MODELS
    seed = spec["seed"]
    model_id = MODELS[model_key]
    data_path = f"{FINAL}/{spec['domain']}_{spec['data_variant']}/train.jsonl"
    run_dir = f"{VOL_RUNS}/{run_name}/{model_key}"
    ckpt_dir = f"{run_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(f"{run_dir}/_MERGED_DONE"):
        print(f"[{run_name}/{model_key}] already complete, skipping")
        return json.dumps({"status": "already_done"})

    set_seed(seed)
    t0 = time.time()
    tok_kw = {"fix_mistral_regex": True} if model_key == "ministral" else {}
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    n_raw = len(rows)
    # Drop rows with an empty prompt or response: they contribute no supervised
    # tokens, and Mistral's chat template rejects empty assistant content.
    rows = [r for r in rows
            if str(r.get("prompt", "")).strip() and str(r.get("response", "")).strip()]
    if len(rows) != n_raw:
        print(f"[{run_name}/{model_key}] dropped {n_raw - len(rows)} empty row(s)")
    print(f"[{run_name}/{model_key}] {len(rows)} rows from {data_path}")
    tmpl_kw = CHAT_TEMPLATE_KWARGS.get(model_key, {})

    def lcp(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    feats, n_trunc = [], 0
    for row in rows:
        msgs = [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
        s_gen = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                        add_generation_prompt=True, **tmpl_kw)
        s_full = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=False, **tmpl_kw)
        cp = lcp(s_gen, s_full)
        prompt_ids = tok(s_gen, add_special_tokens=False)["input_ids"]
        target_ids = tok(s_full[cp:], add_special_tokens=False)["input_ids"]
        full_ids = list(prompt_ids) + list(target_ids)
        labels = [-100] * len(prompt_ids) + list(target_ids)
        if len(full_ids) > MAX_SEQ_LEN:
            full_ids, labels = full_ids[:MAX_SEQ_LEN], labels[:MAX_SEQ_LEN]
            n_trunc += 1
        if all(x == -100 for x in labels):
            continue
        feats.append({"input_ids": full_ids, "labels": labels,
                      "attention_mask": [1] * len(full_ids)})
    ds = Dataset.from_list(feats)
    print(f"[{run_name}/{model_key}] {len(ds)} examples, {n_trunc} truncated")

    cfg = AutoConfig.from_pretrained(model_id)
    arch = cfg.architectures[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        from transformers import AutoModelForCausalLM as cls  # noqa: N813
    model = cls.from_pretrained(model_id, dtype=torch.bfloat16)
    model.config.use_cache = False
    targets = find_lora_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.enable_input_require_grads()

    class VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            vol.commit()

        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                with open(f"{run_dir}/train_log.jsonl", "a") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    targs = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        logging_steps=LOG_STEPS, save_steps=SAVE_STEPS, save_total_limit=2,
        seed=seed, data_seed=seed, gradient_checkpointing=True,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=4)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(
                          tok, padding=True, label_pad_token_id=-100),
                      callbacks=[VolCommit()])
    last_ckpt = get_last_checkpoint(ckpt_dir)
    if last_ckpt:
        print(f"[{run_name}/{model_key}] RESUMING from {last_ckpt}")
    result = trainer.train(resume_from_checkpoint=last_ckpt)

    adapter_dir = f"{run_dir}/adapter"
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    vol.commit()
    del model, trainer
    torch.cuda.empty_cache()
    base = cls.from_pretrained(model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged_dir = f"{run_dir}/merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(model_id).save_pretrained(merged_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] processor save skipped: {e}")

    hours = (time.time() - t0) / 3600
    run_config = {
        "run_name": run_name, "model_key": model_key, "model_id": model_id,
        **spec, "n_train": len(ds), "n_truncated": n_trunc,
        "max_seq_len": MAX_SEQ_LEN,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "quantization": "none (bf16)"},
        "epochs": NUM_EPOCHS, "lr": LEARNING_RATE,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "gpu": GPU_TYPE, "gpu_fallback": "none",
        "train_loss_final": result.training_loss,
        "train_hours": round(hours, 3),
        "est_cost_usd": round(hours * GPU_PRICE_PER_HR, 2),
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    with open(f"{run_dir}/run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    open(f"{run_dir}/_MERGED_DONE", "w").close()
    vol.commit()
    print(f"[{run_name}/{model_key}] DONE loss={result.training_loss:.4f} "
          f"hours={hours:.2f} cost=${hours * GPU_PRICE_PER_HR:.2f}")
    return json.dumps(run_config, default=str)


# ======================================================================
# driver: prepare, then all trainings sequentially, with progress+cost log
# ======================================================================
@app.function(image=cpu_image, volumes={VOL: vol},
              timeout=24 * 3600, cpu=2, memory=4096)  # Modal max; queue is
def train_all():                                       # idempotent — relaunch resumes
    prepare_final.remote()
    vol.reload()                       # pick up dirs created by prepare's container
    os.makedirs(FINAL, exist_ok=True)
    jobs = [(rn, mk) for rn in RUN_ORDER for mk in MODEL_ORDER]
    progress = {"total": len(jobs), "done": [], "cost_usd": 0.0,
                "gpu_hours": 0.0, "started": time.time()}
    for i, (rn, mk) in enumerate(jobs):
        print(f"=== [{i + 1}/{len(jobs)}] {rn}/{mk} ===", flush=True)
        try:
            rc = json.loads(train.remote(rn, mk))
        except Exception as e:  # noqa: BLE001 — log and continue the queue
            rc = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"[FAILED] {rn}/{mk}: {rc['error']}", flush=True)
        progress["done"].append({"run": rn, "model": mk,
                                 "hours": rc.get("train_hours"),
                                 "cost": rc.get("est_cost_usd"),
                                 "error": rc.get("error")})
        progress["cost_usd"] += rc.get("est_cost_usd") or 0
        progress["gpu_hours"] += rc.get("train_hours") or 0
        n_done = len(progress["done"])
        rate = (time.time() - progress["started"]) / n_done
        progress["eta_hours_remaining"] = round(rate * (len(jobs) - n_done) / 3600, 2)
        with open(f"{FINAL}/PROGRESS.json", "w") as f:
            json.dump(progress, f, indent=2)
        vol.commit()
        print(f"[progress] {n_done}/{len(jobs)} gpu_cost=${progress['cost_usd']:.2f} "
              f"eta={progress['eta_hours_remaining']}h", flush=True)
    return progress


@app.function(image=cpu_image, volumes={VOL: vol},
              timeout=24 * 3600, cpu=2, memory=4096)
def train_additions():
    """The additional datasets x 3 models, serial, idempotent.
    Progress: ext/final/PROGRESS_ADD.json."""
    prepare_additions.remote()
    vol.reload()
    os.makedirs(FINAL, exist_ok=True)
    jobs = [(rn, mk) for rn in RUN_ORDER_ADDITIONS for mk in MODEL_ORDER]
    progress = {"total": len(jobs), "done": [], "cost_usd": 0.0,
                "gpu_hours": 0.0, "started": time.time()}
    for i, (rn, mk) in enumerate(jobs):
        print(f"=== add [{i + 1}/{len(jobs)}] {rn}/{mk} ===", flush=True)
        try:
            rc = json.loads(train.remote(rn, mk))
        except Exception as e:  # noqa: BLE001
            rc = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"[FAILED] {rn}/{mk}: {rc['error']}", flush=True)
        progress["done"].append({"run": rn, "model": mk,
                                 "hours": rc.get("train_hours"),
                                 "cost": rc.get("est_cost_usd"),
                                 "error": rc.get("error")})
        progress["cost_usd"] += rc.get("est_cost_usd") or 0
        progress["gpu_hours"] += rc.get("train_hours") or 0
        n_done = len(progress["done"])
        rate = (time.time() - progress["started"]) / n_done
        progress["eta_hours_remaining"] = round(rate * (len(jobs) - n_done) / 3600, 2)
        with open(f"{FINAL}/PROGRESS_ADD.json", "w") as f:
            json.dump(progress, f, indent=2)
        vol.commit()
        print(f"[progress-add] {n_done}/{len(jobs)} "
              f"gpu_cost=${progress['cost_usd']:.2f} "
              f"eta={progress['eta_hours_remaining']}h", flush=True)
    return progress


@app.local_entrypoint()
def launch_all():
    print(json.dumps(train_all.remote(), indent=2)[:2000])


@app.local_entrypoint()
def launch_additions():
    print(json.dumps(train_additions.remote(), indent=2)[:2000])


@app.local_entrypoint()
def prepare_only(force: bool = False):
    print(json.dumps(prepare_final.remote(force), indent=2)[:3000])


@app.local_entrypoint()
def main(run_name: str, model_key: str):
    print(train.remote(run_name, model_key))
