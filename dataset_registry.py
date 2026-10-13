"""Registry of the 12 fine-tuning datasets (4 domains x 3).

Datasets are processed in `priority` order within a domain: the anchor first, then
the two additional datasets. Cross-dataset deduplication (data/build_samples.py)
lets an earlier dataset claim a prompt; a later one must drop it and refill from
the same stratum. `strat_col` is the sub-category field used for proportional
stratified sampling; None means no usable metadata and a derived stratum is used.
"""

# ---------------------------------------------------------------- registry
EXT_DATASETS = {
    "math": [
        {
            "key": "openr1", "hf_id": "open-r1/OpenR1-Math-220k",
            "config": "default", "split": "train",
            "prompt_col": "problem", "response_col": "solution",
            "strat_col": "problem_type", "id_col": "uuid",
            "role": "anchor", "priority": 0,
        },
        {
            "key": "nemotron", "hf_id": "nvidia/Nemotron-SFT-Math-v3",
            "config": "default", "split": "train",
            "prompt_col": "problem", "response_col": None,   # answer lives in messages
            "messages_col": "messages",
            "strat_col": "data_source", "id_col": "uuid",
            "role": "addition", "priority": 1,
        },
        {
            "key": "ultradata", "hf_id": "openbmb/UltraData-SFT-2605",
            "config": None, "split": "train",
            # English math files only, direct-answer (no_think) variant.
            "data_files": "data/no_think/Math/*.jsonl",
            "prompt_col": None, "response_col": None,   # messages-style; resolved at scan
            "messages_col": "messages",
            "strat_col": None, "id_col": None,
            "role": "addition", "priority": 2,
            "gated": True,
            "derived_stratum": "prompt_topic_heuristic",
        },
    ],
    "code": [
        {
            "key": "ocr", "hf_id": "nvidia/OpenCodeReasoning",
            "config": "split_0", "split": "split_0",
            "prompt_col": "input", "response_col": "solution",
            "strat_col": "source", "id_col": "id",
            "role": "anchor", "priority": 0,
        },
        {
            "key": "xcoder", "hf_id": "IIGroup/X-Coder-SFT-376k",
            "config": "default", "split": "unique_prompt_202k",
            "prompt_col": "query", "response_col": "response",
            "strat_col": None, "id_col": None,
            "role": "addition", "priority": 1,
            "note": "4 splits (hybrid_376k / multiple_solution_218k / "
                    "unique_prompt_202k / verified_90k) are OVERLAPPING VIEWS of the "
                    "same pool, not disjoint shards. unique_prompt_202k chosen: one "
                    "row per distinct prompt, so within-dataset duplication is "
                    "already removed. No sub-category column -> derived stratum.",
            "derived_stratum": "prompt_topic_heuristic",
        },
        {
            "key": "starcoder_py", "hf_id": "OLMo-Coding/starcoder-python-instruct",
            "config": "default", "split": "train",
            "prompt_col": "instruction", "response_col": "text",
            "strat_col": "metadata.extension", "id_col": "id",
            "role": "addition", "priority": 2,
            "note": "1.54M rows. `text` is a raw source file, not a chat response — "
                    "response style differs from the other code sets; flagged as an "
                    "format confound noted in the paper.",
        },
    ],
    "medical": [
        {
            "key": "medo1", "hf_id": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "config": "en", "split": "train",
            "prompt_col": "Question", "response_col": "Response",
            "strat_col": None, "id_col": None,
            "role": "anchor", "priority": 0,
            "derived_stratum": "prompt_topic_heuristic",
        },
        {
            "key": "medcase", "hf_id": "zou-lab/MedCaseReasoning",
            "config": "default", "split": "train",
            "prompt_col": "case_prompt", "response_col": "final_diagnosis",
            "strat_col": "journal", "id_col": "pmcid",
            "role": "addition", "priority": 1,
            "note": "ONLY ~13k train rows (14,489 across train/val/test). Cannot "
                    "supply 5,000 unique rows if dedup drops are heavy — flag, "
                    "do not pad.",
        },
        {
            "key": "mediflow", "hf_id": "microsoft/mediflow",
            "config": "default", "split": "mediflow",
            "prompt_col": "instruction", "response_col": "output",
            "input_col": "input",
            "strat_col": "task_type", "id_col": "example_id",
            "role": "addition", "priority": 2,
            # direct-answer rows only (no chain-of-thought traces)
            "row_filter": {"col": "has_cot", "equals": False},
            "note": "2.69M rows; rich strata (task_type / input_data / "
                    "difficulty_level). Use the `mediflow` split, not mediflow_dpo.",
        },
    ],
    "safety": [
        {
            # Scanned and sampled like the others; data/build_aegis_safe_sample.py then keeps
            # only the safe-labelled responses (refilling to the token budget).
            "key": "aegis", "hf_id": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
            "config": "default", "split": "train",
            "prompt_col": "prompt", "response_col": "response",
            "strat_col": "violated_categories", "id_col": "id",
            "role": "anchor", "priority": 0,
            # Aegis contains prompt-only annotation rows (response null/empty) —
            # unusable as SFT targets; ~58% of an unfiltered sample. Filter at
            # scan time so strata proportions + pools cover usable rows only.
            "row_filter": {"col": "response", "nonempty": True},
            "note": "33,416 rows. Human-annotated, independent lineage from the "
                    "two AI2 sets -> the right anchor for this domain.",
        },
        {
            "key": "wildjailbreak", "hf_id": "allenai/wildjailbreak",
            "config": "train", "split": "train",
            "prompt_col": "adversarial", "response_col": "completion",
            "strat_col": "data_type", "id_col": None,
            "role": "addition", "priority": 1,
            "gated": True,
            "special": "wjb_prompt",   # vanilla rows leave `adversarial` empty
            "note": "data_type strata: vanilla/adversarial x benign/harmful. "
                    "Prompt = adversarial if non-empty else vanilla.",
        },
        {
            # Intermediate sample only: the PKU-SafeRLHF training set (pku_safe below) is built
            # from it by data/build_pku_safe_sample.py. Not trained on.
            "key": "pku_saferlhf", "hf_id": "PKU-Alignment/PKU-SafeRLHF",
            "config": "default", "split": "train",
            "prompt_col": "prompt", "response_col": "response",  # produced by pku_extract
            "strat_col": None, "id_col": None,
            "role": "intermediate", "priority": 2,
            "special": "pku_safer_response",
            "note": "Preference dataset, not SFT: each row has two responses with "
                    "safety labels. SFT target = the response indexed by "
                    "safer_response_id. Stratum = severity level + primary harm "
                    "category of that chosen response ('safe|none' when unharmful).",
        },
        {
            # PKU-SafeRLHF training set: same extraction, but rows whose chosen
            # response is labelled unsafe are dropped and refilled from the safe pool.
            "key": "pku_safe", "hf_id": "PKU-Alignment/PKU-SafeRLHF",
            "config": "default", "split": "train",
            "prompt_col": "prompt", "response_col": "response",
            "strat_col": None, "id_col": None,
            "role": "addition", "priority": 3,
            "special": "pku_safer_response_safe_only",
        },
    ],
}

TARGET_N = 5000          # nominal per-dataset target before token matching
DOMAINS = list(EXT_DATASETS)


def flat():
    for domain, specs in EXT_DATASETS.items():
        for s in specs:
            yield domain, s
