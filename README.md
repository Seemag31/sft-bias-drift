# Less Abstention, More Bias?

Code and results for *Less Abstention, More Bias? Demographic Bias Drift After Supervised
Fine-Tuning*.

We fine-tune three instruction-tuned models (Qwen3.5-9B, Gemma-4-12B,
Ministral-3-8B-Instruct) with LoRA on 12 datasets across four domains (math, code,
medical, safety) under a matched 1.0-million-token budget, and measure demographic bias
before and after on the ambiguous fold of BBQ.

| Experiment | Question | Scripts |
|---|---|---|
| 1. Standard SFT | Does fine-tuning change bias, and where? | `training/train_lora.py`, `evaluation/eval_bbq.py`, `analysis/main_results.py` |
| 2. Abstention-augmented SFT | Does keeping ~10% abstention data in the training mix prevent it? | `training/train_lora.py` (`_idk` runs), `analysis/main_results.py` |
| 3. Inference-time deliberation | Does a think-before-answer prompt reduce it without retraining? | `evaluation/eval_deliberation.py`, `analysis/deliberation_table.py` |

## Layout

```
config.py                  models, hyperparameters, seeds, run matrix
dataset_registry.py        the 12 fine-tuning datasets
data/                      Modal jobs that build the training data
  download_benchmarks.py     BBQ, GSM8K, HumanEval, MedQA, Aegis test split
  scan_datasets.py           one streaming pass per dataset: strata, hashes, BBQ contamination
  build_samples.py           cross-dataset dedup, stratified token-matched samples
  build_aegis_safe_sample.py Aegis 2.0 sample restricted to safe-labelled responses
  build_pku_safe_sample.py   PKU-SafeRLHF sample restricted to safe-labelled responses
training/train_lora.py     training files (standard and abstention-augmented) and LoRA training
evaluation/                Modal jobs that evaluate every run
  eval_bbq.py                BBQ, free generation or guided choice
  eval_capability.py         target-task benchmarks
  eval_deliberation.py       direct vs think-before-answer on the 1,100-item subset
  download_results.py        mirror predictions and training files into artifacts/
analysis/                  predictions -> results/tables/*.csv
figures/                   results/tables -> results/figures/*.pdf
results/tables, results/figures
```

## Reproducing the results

Remote jobs run on [Modal](https://modal.com) with one GPU container at a time and commit
to a persistent volume. A Modal secret `huggingface-secret` holding `HF_TOKEN` is required
(some datasets and models are gated).

```bash
pip install -r requirements.txt

# 1. data (CPU)
python -m modal run data/download_benchmarks.py
python -m modal run data/scan_datasets.py
python -m modal run --detach data/build_samples.py
python -m modal run data/build_pku_safe_sample.py
python -m modal run data/build_aegis_safe_sample.py

# 2. training (60 runs, ~0.3 GPU-h each)
python -m modal deploy training/train_lora.py
python -m modal run --detach training/train_lora.py::launch_all         # anchors: seed 1, seed 2, abstention-augmented
python -m modal run --detach training/train_lora.py::launch_additions   # the other datasets

# 3. evaluation
python -m modal deploy evaluation/eval_bbq.py
python -m modal run --detach evaluation/eval_bbq.py::launch_all
python -m modal run --detach evaluation/eval_bbq.py::launch_additions
python -m modal run --detach evaluation/eval_capability.py
python -m modal run --detach evaluation/eval_deliberation.py
python evaluation/download_results.py

# 4. tables and figures (local)
for f in main_results per_category abstention_correlation decomposition demographic_content \
         decomposition_by_category context_sensitivity decoding_regime switch_counts deliberation_table capability_table; do
  python analysis/$f.py
done
for f in figures/fig_*.py; do python "$f"; done
```

## Design

Each dataset is sampled to 1.0 million training tokens (Qwen3.5-9B tokenizer, chat format,
2,048-token cap) with stratification preserved and cross-dataset near-duplicates removed.
Aegis 2.0 and PKU-SafeRLHF targets are the safe-labelled responses. Every run is bf16 LoRA on
all linear projections (r = 16, alpha = 32, dropout 0.05, lr 2e-4, cosine schedule with 3%
warmup, effective batch 16, one epoch), with prompt tokens masked from the loss.

Bias is the BBQ ambiguous-context score s_AMB = (n_bias - n_anti) / N over all 15,686
ambiguous items, evaluated with one user turn through each model's chat template and greedy
decoding. Math and medical runs are scored under free generation; code and safety runs under
guided choice (A/B/C), where fine-tuning otherwise breaks the answer format. z statistics use a
paired item bootstrap (1,000 resamples, seed 42).

Run names: `ext_<domain>_s1` (standard SFT, seed 42), `ext_<domain>_s2` (seed 1337),
`ext_<domain>_idk` (abstention-augmented), `ext_<domain>_<dataset>` (additional datasets);
a `__guided` suffix marks guided-choice evaluation.

## Paper tables and figures

| Paper | Script | File in `results/` |
|---|---|---|
| Tables 1, 3, 8 (36 conditions; fold change, z, abstention) | `analysis/main_results.py` | `tables/table03_full_results.csv` |
| Table 4 (second seed) | `analysis/main_results.py` | `tables/table04_seed_replication.csv` |
| Tables 5, 6 (category changes) | `analysis/per_category.py` | `tables/table05_06_category_changes.csv` |
| Table 7 (abstention-bias correlation) | `analysis/abstention_correlation.py` | `tables/table07_abstention_correlation.csv` |
| Table 9 (bias decomposition, disambiguated check) | `analysis/decomposition.py` | `tables/table09_decomposition.csv` |
| Table 10 (decomposition by category) | `analysis/decomposition_by_category.py` | `tables/table10_decomposition_category.csv` |
| Table 11 (training-data demographic content) | `analysis/demographic_content.py` | `tables/table11_demographic_density.csv`, `table11_pooled_test.csv` |
| Table 12 (category fold change, standard vs augmented) | `analysis/per_category.py` | `tables/table12_category_fold_change.csv` |
| Table 13 (matched conditions, Experiment 2) | `analysis/main_results.py` | `tables/table13_abstention_control.csv` |
| Table 14 (context sensitivity) | `analysis/context_sensitivity.py` | `tables/table14_context_sensitivity.csv` |
| Table 15 (deliberation) | `analysis/deliberation_table.py` | `tables/table15_deliberation.csv` |
| Table 16 (target-task capability) | `analysis/capability_table.py` | `tables/table16_capability.csv` |
| Table 17, F.1 (decoding regime, invalid responses) | `analysis/decoding_regime.py` | `tables/table17_decoding_regime.csv`, `table17_invalid_summary.csv` |
| Appendix C.4 (item-level switch counts) | `analysis/switch_counts.py` | `tables/c4_switch_counts.csv` |
| Figure 3a | `figures/fig_category_domain_fc.py` | `figures/fig03a_category_domain.pdf` |
| Figure 3b | `figures/fig_bias_abstention.py` | `figures/fig03b_bias_vs_abstention.pdf` |
| Figure 4 | `figures/fig_abstention_control.py` | `figures/fig04_abstention_control_<model>.pdf` |
| Figure 5 | `figures/fig_deliberation.py` | `figures/fig05_deliberation_<model>.pdf` |
| Figure 6 | `figures/fig_bias_shift.py` | `figures/fig06_bias_shift.pdf` |
| Figure 7 | `figures/fig_abstention_category.py` | `figures/fig07_category_abstention.pdf` |

`tables/per_category.csv` holds the per-category scores behind Tables 5, 6, 11, 12 and
Figures 3a and 7.

## What is included

The full pipeline (data preparation, training, evaluation, analysis, figures) and the results
it produced. Model checkpoints, per-item predictions, and training data are not included:
they are rebuilt from the public datasets by running the pipeline, and the analysis scripts
read them from `artifacts/`.
