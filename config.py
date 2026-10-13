"""Single source of truth for models, hyperparameters, seeds, BBQ, and the run matrix.

Every script in data/, training/, evaluation/, and analysis/ imports from here, so
nothing about the experimental design is defined twice.
"""

# ---------------------------------------------------------------- models
MODELS = {
    "qwen": "Qwen/Qwen3.5-9B",
    "gemma": "google/gemma-4-12B-it",
    # Mistral's official BF16 release of the same model (the default repo ships
    # FP8 weights, which would break "standard non-quantized LoRA in bf16").
    "ministral": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
}
MODEL_ORDER = ["qwen", "gemma", "ministral"]

# Chat-template kwargs applied identically in training and evaluation.
# Qwen3.5 opens a <think> block by default; enable_thinking=False makes the
# template emit a pre-closed empty block so the model answers directly.
CHAT_TEMPLATE_KWARGS = {
    "qwen": {"enable_thinking": False},
    "gemma": {},
    "ministral": {},
}

# ---------------------------------------------------------------- seeds
SEED_SAMPLING = 42        # data sampling
SEED_TRAIN = 42           # seed 1 (all main-grid runs)
SEED_TRAIN_REPLICATION = 1337   # seed 2 (Appendix: seed replication)
SEED_GENERATION = 1234    # vLLM sampling seed (greedy, so only tie-breaking)

# ---------------------------------------------------------------- LoRA / training
TOKEN_BUDGET = 1_000_000  # training tokens per run (Qwen3.5 tokenizer, chat-formatted)
MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4            # effective batch 16
NUM_EPOCHS = 1
SAVE_STEPS = 50
LOG_STEPS = 10
GPU_TYPE = "RTX-PRO-6000"
GPU_PRICE_PER_HR = 3.03   # for cost logging only

# ---------------------------------------------------------------- BBQ
BBQ_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Religion", "SES",
    "Sexual_orientation",
]
BBQ_RAW_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{cat}.jsonl"

# Domain benchmarks used for the target-task capability check (Table 15).
DOMAIN_BENCHMARKS = {
    "math": "gsm8k",          # GSM8K test (1,319), exact-match numeric
    "code": "humaneval",      # HumanEval (164), pass@1, sandboxed execution
    "medical": "medqa",       # MedQA USMLE 4-option test (1,273), letter accuracy
    "safety": "aegis_test",   # Aegis 2.0 test split, safe/unsafe classification
}

# ---------------------------------------------------------------- Modal volume layout
VOLUME_NAME = "sft-bias-pilot-artifacts"
HF_CACHE_VOLUME = "huggingface-cache"
HF_SECRET_NAME = "huggingface-secret"   # Modal secret holding HF_TOKEN
VOL = "/vol"
VOL_BBQ = f"{VOL}/data/bbq"
VOL_BENCH = f"{VOL}/data/benchmarks"
VOL_RUNS = f"{VOL}/runs"                # runs/<run_name>/<model>/{adapter,merged,...}
VOL_EVAL = f"{VOL}/eval"                # eval/bbq/..., eval/domain_ext/..., eval/task8/...
VOL_EXT = f"{VOL}/ext"                  # ext/{scan,pool,dedup,sample,final}/...
VOL_FINAL = f"{VOL_EXT}/final"          # final/<run>/train.jsonl + progress files

# ---------------------------------------------------------------- run matrix
# Anchor dataset per domain (Experiments 2 and 3 use these four). The Aegis 2.0 and
# PKU-SafeRLHF training targets are safe-labelled responses only (see data/).
ANCHORS = {"math": "openr1", "code": "ocr", "medical": "medo1", "safety": "aegis"}

# Abstention-augmented arm (Experiment 2): ~10% of the anchor's training tokens
# are REPLACED by CoCoNot examples whose correct response is to abstain.
IDK_FRAC = 0.10
IDK_SOURCE = "allenai/coconot"
IDK_CATEGORIES = ("incomplete", "indeterminate")   # substring match on CoCoNot category

# Run names are stable identifiers on the volume:
#   ext_<domain>_s1            standard SFT on the anchor, seed 1   (main grid)
#   ext_<domain>_s2            same data, seed 2                    (seed replication)
#   ext_<domain>_idk           abstention-augmented, seed 1         (Experiment 2)
#   ext_<domain>_<dataset>     standard SFT on an additional dataset (main grid)
RUNS = {}
for _d, _a in ANCHORS.items():
    RUNS[f"ext_{_d}_s1"] = {"domain": _d, "dataset": _a, "seed": SEED_TRAIN, "data_variant": "normal"}
    RUNS[f"ext_{_d}_s2"] = {"domain": _d, "dataset": _a, "seed": SEED_TRAIN_REPLICATION, "data_variant": "normal"}
    RUNS[f"ext_{_d}_idk"] = {"domain": _d, "dataset": _a, "seed": SEED_TRAIN, "data_variant": "idk"}

ADDITIONS = {
    "math": ["nemotron", "ultradata"],
    "code": ["xcoder", "starcoder_py"],
    "medical": ["medcase", "mediflow"],
    # PKU-SafeRLHF targets are the safe-labelled responses only (data/build_pku_safe_sample.py).
    "safety": ["wildjailbreak", "pku_safe"],
}
for _d, _keys in ADDITIONS.items():
    for _k in _keys:
        RUNS[f"ext_{_d}_{_k}"] = {"domain": _d, "dataset": _k, "seed": SEED_TRAIN, "data_variant": _k}

RUN_ORDER = ([f"ext_{d}_s1" for d in ANCHORS] + [f"ext_{d}_s2" for d in ANCHORS]
             + [f"ext_{d}_idk" for d in ANCHORS])
RUN_ORDER_ADDITIONS = [f"ext_{d}_{k}" for d in ADDITIONS for k in ADDITIONS[d]]

# The 12 datasets of the main grid, per domain, with the run that produces each cell.
MAIN_GRID = {
    "math": {"openr1": "ext_math_s1", "nemotron": "ext_math_nemotron", "ultradata": "ext_math_ultradata"},
    "code": {"ocr": "ext_code_s1", "xcoder": "ext_code_xcoder", "starcoder_py": "ext_code_starcoder_py"},
    "medical": {"medo1": "ext_medical_s1", "medcase": "ext_medical_medcase", "mediflow": "ext_medical_mediflow"},
    "safety": {"aegis": "ext_safety_s1", "wildjailbreak": "ext_safety_wildjailbreak", "pku_safe": "ext_safety_pku_safe"},
}

# Decoding regime used for the main grid: free generation where invalid-response
# rates stay near zero, guided choice (A/B/C) where fine-tuning breaks the answer format.
MAIN_REGIME = {"math": "free", "medical": "free", "code": "guided", "safety": "guided"}
