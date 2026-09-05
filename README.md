# Less Abstention, More Bias?

Code and results for the paper *Less Abstention, More Bias? Demographic Bias Drift
After Supervised Fine-Tuning* (NeurIPS 2026 submission).

We fine-tune three instruction-tuned models (Qwen3.5-9B, Gemma-4-12B, Ministral-3-8B)
with LoRA on 12 datasets across four domains (math, code, medical, safety) under a
matched 1.0-million-token budget, and measure demographic bias before and after on the
ambiguous fold of BBQ. Three experiments:

| Experiment | Question | Scripts |
|---|---|---|
| 1. Standard SFT | Does fine-tuning change bias, and where? | `training/train_lora.py`, `evaluation/eval_bbq.py`, `analysis/results_grid.py` |
| 2. Abstention-augmented SFT | Does keeping ~10% abstention data in the training mix prevent it? | same training script (`_idk` runs), `analysis/results_grid.py` |
| 3. Inference-time deliberation | Does a think-before-answer prompt reduce it without retraining? | `evaluation/eval_deliberation.py`, `analysis/deliberation_table.py` |

## Layout

```
config.py                 models, hyperparameters, seeds, BBQ, volume paths, run matrix
dataset_registry.py       the 12 fine-tuning datasets (+ PKU safe-only control arm)
data/                     Modal jobs that build the training data on a persistent volume
  download_benchmarks.py    BBQ + GSM8K / HumanEval / MedQA / Aegis test
  scan_datasets.py          one streaming pass per dataset: strata, hashes, BBQ contamination
  build_samples.py          cross-dataset dedup + stratified, token-matched samples
  build_pku_safe_sample.py  safe-only PKU-SafeRLHF sample
training/train_lora.py    train files (standard + abstention-augmented) and LoRA training
evaluation/               Modal jobs that evaluate every run
  eval_bbq.py               BBQ, free generation or guided choice
  eval_capability.py        target-task benchmarks (Table 2)
  eval_deliberation.py      Experiment 3 design on the 1,100-item subset
  download_results.py       mirror predictions from the volume into artifacts/
analysis/                 local scripts: predictions -> results/tables/*.csv
figures/                  local scripts: results/tables -> results/figures/*.pdf
results/tables            the CSVs behind every number in the paper
results/figures           the figures included in the paper
```

## Reproducing the paper

Everything remote runs on [Modal](https://modal.com) with one GPU container at a time
(`max_containers=1`) and commits to a persistent volume, so jobs survive a dropped client.
Deploy each app once (`python -m modal deploy <script>`), then launch with `modal run`.
A Modal secret named `huggingface-secret` holding `HF_TOKEN` is required (three datasets
and two models are gated on Hugging Face).

```bash
pip install -r requirements.txt

# 1. data (CPU jobs, ~1 h)
python -m modal run data/download_benchmarks.py
python -m modal run data/scan_datasets.py
python -m modal run --detach data/build_samples.py
python -m modal run data/build_pku_safe_sample.py

# 2. training (36 + 12 runs, ~0.3 GPU-h each)
python -m modal deploy training/train_lora.py
python -m modal run --detach training/train_lora.py::launch_all         # anchors: seed 1, seed 2, abstention-augmented
python -m modal run --detach training/train_lora.py::launch_additions   # the other 9 datasets

# 3. evaluation
python -m modal deploy evaluation/eval_bbq.py
python -m modal run --detach evaluation/eval_bbq.py::launch_all
python -m modal run --detach evaluation/eval_bbq.py::launch_additions
python -m modal run --detach evaluation/eval_capability.py
python -m modal run --detach evaluation/eval_deliberation.py
python evaluation/download_results.py

# 4. tables and figures (local, a few minutes)
python analysis/results_grid.py          # main_grid, results_grid, seed_replication, abstention_control
python analysis/bootstrap_z.py           # standardized_drift (SE and z per cell)
python analysis/per_category.py          # per_category
python analysis/abstention_correlation.py
python analysis/group_direction.py
python analysis/pku_safe_control.py
python analysis/capability_table.py
python analysis/deliberation_table.py
for f in figures/fig_*.py; do python "$f"; done
```

Total compute for the paper was about 36 GPU-hours on one RTX PRO 6000.

## Design in one paragraph

Each dataset is sampled to 1.0 million training tokens (Qwen3.5-9B tokenizer, chat
format, 2,048-token cap) with stratification preserved and cross-dataset near-duplicates
removed. Every run is bf16 LoRA on all linear projections (r = 16, alpha = 32, dropout
0.05, lr 2e-4, cosine with 3% warmup, effective batch 16, one epoch), prompt tokens
masked from the loss. Bias is the BBQ ambiguous-context score
s_AMB = (n_bias - n_anti) / N over all 15,686 ambiguous items, evaluated with one user
turn through each model's chat template and greedy decoding. Math and medical runs are
scored under free generation; code and safety runs under guided choice (A/B/C), where
fine-tuning otherwise breaks the answer format. Standard errors come from a paired item
bootstrap (1,000 resamples, seed 42).

## Run names

Run names are stable identifiers on the volume and in `results/tables`:

| Run | Meaning |
|---|---|
| `ext_<domain>_s1` | standard SFT on the domain's anchor dataset, seed 42 |
| `ext_<domain>_s2` | same data, seed 1337 (seed replication) |
| `ext_<domain>_idk` | abstention-augmented SFT (Experiment 2) |
| `ext_<domain>_<dataset>` | standard SFT on an additional dataset |
| `<run>__guided` | the same run evaluated with guided-choice decoding |

Anchors: OpenR1-Math (`openr1`), OpenCodeReasoning (`ocr`), medical-o1 (`medo1`),
Aegis 2.0 (`aegis`). The PKU-SafeRLHF cell in the paper is the safe-only run
(`pku_safe`); `pku_saferlhf` (the "safer of the pair" response, 44% unsafe-labelled)
is kept as a control (`analysis/pku_safe_control.py`).

## Results tables

| File | Contents |
|---|---|
| `main_grid.csv` | the 36 conditions: s_AMB before/after, delta, fold change, abstention, invalid rate |
| `standardized_drift.csv` | bootstrap SE and z per condition (Table 2 markers) |
| `results_grid.csv` | every run x regime, including seed-2 and abstention-augmented runs |
| `per_category.csv` | the same per BBQ category (Figure 2, category tables) |
| `seed_replication.csv` | seed 1 vs seed 2 for the four anchors |
| `abstention_control.csv` | Experiment 2 paired results |
| `abstention_correlation.csv` | Spearman rho between delta abstention and delta s_AMB |
| `group_direction.csv` | which group is named after fine-tuning, by category and polarity |
| `pku_safe_control.csv` | safe-only vs safer-response PKU |
| `capability.csv` | GSM8K / HumanEval / MedQA / Aegis-test accuracy before and after |
| `deliberation_2x2.csv` | Experiment 3 cells on the 1,100-item subset |

Note: This repository contains the full pipeline (data preparation, training, evaluation,
analysis, figures) and the results it produced (`results/`). It does not contain model
checkpoints or per-item prediction files: reproducing the tables from scratch means
running the pipeline, which builds the training data from public Hugging Face datasets,
fetches BBQ from the official nyu-mll release, and writes predictions to `artifacts/`
(layout in `evaluation/download_results.py`). The analysis scripts then read from
`artifacts/` and regenerate everything under `results/`.
