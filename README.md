# Test-Time Compute Scaling for Reasoning Fine-Tuned Models

**Studying how inference accuracy on MATH scales with compute budget and inference strategy**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/Tracked%20with-W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

_Process Reward Model training and test-time scaling experiments with_ `Qwen2.5-Math-Instruct` _on the MATH benchmark._

[Overview](#-overview) · [Inference Methods](#-inference-methods) · [PRM](#-process-reward-model) · [Results](#-results) · [Research Questions](#-research-questions) · [Usage](#-usage) · [References](#-references)

---

## 📌 Overview

Recent work has shown that scaling inference compute — rather than training compute — can substantially improve the accuracy of language models on reasoning tasks [[1]](#-references)[[2]](#-references). The key ingredient for the most powerful test-time strategies is a **Process Reward Model (PRM)**: a model that assigns a correctness score to each intermediate reasoning step, enabling search over the space of reasoning traces.

This project studies the **scaling of accuracy on the [MATH dataset](https://github.com/hendrycks/math) [[8]](#-references) as a function of inference compute** for small open language models, using [`Qwen/Qwen2.5-Math-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) [[9]](#-references) as the generator. We compare four inference strategies at varying rollout budgets $N$:

| Method | PRM Required | Aggregation Level |
|---|---|---|
| **Majority Vote** | No | Answer |
| **Vanilla Best-of-N** | Yes | Trace (last-step score) |
| **Weighted Best-of-N** | Yes | Answer (sum of last-step scores) |
| **PRM-Guided Beam Search** | Yes | Trace (step-level pruning) |

We train two PRM variants to disentangle the effect of training data provenance:

| PRM Checkpoint | Base Model | Training Data Generator |
|---|---|---|
| `PRM_1.5B_Train` | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Math-1.5B-Instruct |
| `PRM_1p5B_7B_Train` | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Math-7B-Instruct |

This setup lets us ask: does a smaller PRM trained on data from a larger model offer a cheaper path to improving inference for that larger model?

---

## 🔬 Inference Methods

### Majority Vote (Self-Consistency)

For each problem, $N$ independent reasoning traces are sampled from the generator LLM. The final answer is determined by plurality vote over the normalised extracted answers [[3]](#-references):

$$\hat{a} = \underset{a}{\arg\max} \sum_{i=1}^{N} \mathbb{1}[\text{normalize}(a_i) = a]$$

No reward model is required. This is the baseline against which PRM-based methods are compared.

### Vanilla Best-of-N

Each of the $N$ rollouts is scored by the PRM, which assigns a probability $p_s \in [0,1]$ to each reasoning step $s$. The rollout with the highest **last-step PRM score** is selected as the answer:

$$\hat{a} = a_{i^*}, \quad i^* = \underset{i}{\arg\max}\; p^{(i)}_{\text{last}}$$

This uses the PRM as a verifier but discards answer-level diversity.

### Weighted Best-of-N

Instead of selecting a single best rollout, per-answer scores are **aggregated** across all rollouts sharing the same normalised answer. This combines the diversity benefit of majority vote with the discriminative power of the PRM [[1]](#-references):

$$\hat{a} = \underset{a}{\arg\max} \sum_{\{i\,:\,a_i = a\}} p^{(i)}_{\text{last}}$$

### PRM-Guided Beam Search

Rather than scoring complete rollouts post-hoc, beam search uses the PRM to **prune the search tree at each reasoning step** [[2]](#-references). Given a budget of $N$ total step expansions per depth and $M$ proposals per beam:

- Maintain $B = N / M$ active beams
- At each depth: each beam proposes $M$ next-step continuations via the LLM ($N$ candidates total)
- The PRM scores each candidate; only the top $B$ beams survive to the next depth
- Generation halts when a beam produces a `\boxed{}` expression or `max_steps` is reached

This explores a directed tree of reasoning paths and focuses compute on promising prefixes.

---

## 🏋️ Process Reward Model

### Architecture

The PRM is built on top of a frozen [`Qwen2.5-Math-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) backbone with a single linear classification head of dimension 1 appended to the last hidden layer. The base model weights are frozen and only the head is trained, keeping the parameter count for the trained component minimal.

Given a tokenised sequence of a prompt followed by $K$ reasoning steps separated by a special step separator token `\n<step>\n`, the PRM predicts a **step-level binary label** at the position of each separator token:

$$p_k = \sigma\!\left(W \cdot h_{\text{sep}_k}\right), \quad k = 1,\ldots,K$$

where $h_{\text{sep}_k}$ is the hidden state at the $k$-th separator position and $W$ is the learned head. Training uses binary cross-entropy loss on these positions only; all other token positions are masked.

### Training Data Generation

PRM training data is generated via **Monte Carlo (MC) rollout estimation** on a random 1000-sample subset of the MATH training set. The procedure follows [[4]](#-references)[[5]](#-references):

**Step 1 — Rollout generation:** For each problem, $G = 8$ full reasoning traces are sampled from the generator LLM using vLLM.

**Step 2 — Step-level MC evaluation:** For each step $k$ in each trace, $G_\text{MC} = 8$ continuations are sampled from that step onwards. A step is labelled **positive** ($y_k = 1$) if at least one continuation leads to a correct final answer, and **negative** ($y_k = 0$) otherwise.

This produces a dataset of (prompt, steps, step-labels) triples that approximate the **per-step outcome probability** under the generator policy — the quantity the PRM is trained to predict.

Two datasets are generated with different generator LLMs, yielding the two PRM checkpoints described in the [Overview](#-overview).

### Training Details

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen2.5-Math-1.5B-Instruct` |
| Learning rate | `2e-5` |
| Batch size | 1 + gradient accumulation 16 |
| Epochs | 3 |
| Warmup ratio | 0.1 |
| Max token length | 2048 |
| Optimizer | AdamW |
| Precision | bfloat16 |

---

## 📊 Results

### Accuracy vs. Rollout Budget

The plots below show the test accuracy on the MATH benchmark as a function of the number of rollouts $N$ for all four inference methods and both PRM variants.

<div align="center">
<!-- TODO: insert accuracy-vs-rollouts plot -->
<img src="Figs/accuracy_vs_rollouts.png" width="90%" alt="Accuracy vs rollout budget for all methods"/>
</div>

### PRM Data Source: 1.5B- vs. 7B-Generated Training Data

<div align="center">
<!-- TODO: insert 1.5B vs 7B PRM comparison plot -->
<img src="Figs/prm_data_source_comparison.png" width="80%" alt="Comparison of PRM trained on 1.5B vs 7B generated data"/>
</div>

### Accuracy by Problem Difficulty

The MATH dataset assigns difficulty levels 1–5 to each problem. The plots below break down method performance by difficulty level to assess whether harder problems benefit more from PRM guidance.

<div align="center">
<!-- TODO: insert per-difficulty breakdown plot -->
<img src="Figs/accuracy_by_difficulty.png" width="90%" alt="Accuracy by MATH difficulty level"/>
</div>

### Summary Table

Results at $N = 8$ rollouts on the MATH test set:

| Method | PRM | Accuracy |
|---|---|---|
| Majority Vote | — | |
| Vanilla Best-of-N | PRM (1.5B data) | |
| Weighted Best-of-N | PRM (1.5B data) | |
| Beam Search | PRM (1.5B data) | |
| Vanilla Best-of-N | PRM (7B data) | |
| Weighted Best-of-N | PRM (7B data) | |
| Beam Search | PRM (7B data) | |

---

## ❓ Research Questions

This project is designed to answer three concrete questions:

**i) Does the source of PRM training data matter?**

The PRM architecture and base model are identical in both checkpoints; only the generator LLM used to produce the MC rollout data differs. Comparing `PRM_1.5B_Train` (1.5B-generated data) vs. `PRM_1p5B_7B_Train` (7B-generated data) isolates the effect of data quality on PRM-guided inference for the 1.5B generator.

**ii) Can a small PRM trained on large-model data improve inference for the large model?**

A 1.5B PRM is much cheaper to run than a 7B PRM during inference. If `PRM_1p5B_7B_Train` improves accuracy when used to guide a 7B generator, this provides a practical strategy for test-time compute scaling: use a smaller verifier trained on the target model's rollouts.

**iii) Does inference method performance depend on problem difficulty?**

Majority vote may be sufficient for easy problems where the generator rarely makes mistakes, while PRM-guided search may only pay off for harder problems where diverse traces are needed. We stratify results by MATH difficulty level (1–5) to test this hypothesis.

---

## ⚙️ Installation

```bash
git clone https://github.com/maxruhdorfer/Test-Time-Compute-For-Reasoning-Fine-Tuned-Models.git
cd Test-Time-Compute-For-Reasoning-Fine-Tuned-Models

pip install -r requirements.txt
```

> **Hardware note:** Experiments were run on a single A100 80GB GPU. Reduce `--gpu_memory_utilization` for smaller GPUs.

---

## 🚀 Usage

### Generate PRM Training Data

```bash
python generate_PRM_data.py \
    --model_id Qwen/Qwen2.5-Math-1.5B-Instruct \
    --train_dataset data/MATH/train.jsonl \
    --train_samples 1000 \
    --rollouts 8 \
    --rollouts_MC 8 \
    --output data/PRM_Train/1.5B/
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | Generator LLM for rollouts and MC continuations |
| `--train_samples` | `1000` | Number of problems sampled from the MATH training set |
| `--rollouts` | `8` | Full rollouts generated per problem ($G$) |
| `--rollouts_MC` | `8` | MC continuations sampled per step ($G_\text{MC}$) |
| `--max_tokens` | `2048` | Maximum tokens per rollout |
| `--sampling_temperature` | `1.0` | Sampling temperature |

### Train a PRM

```bash
python train_PRM.py \
    --model_id Qwen/Qwen2.5-Math-1.5B-Instruct \
    --train_data_path data/PRM_Train/1.5B/PRM_data.jsonl \
    --epochs 3 \
    --lr 2e-5 \
    --gradient_accumulation_steps 16 \
    --run_name my_prm_run
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | Base model for the PRM |
| `--train_data_path` | `data/PRM_Train/7B/PRM_7B_data.jsonl` | Path to generated PRM training data |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-5` | AdamW learning rate |
| `--gradient_accumulation_steps` | `16` | Gradient accumulation steps |
| `--batch_size` | `1` | Per-step batch size |
| `--max_tokens` | `2048` | Maximum token length |
| `--val_fraction` | `0.1` | Fraction of data held out for validation |
| `--warmup_ratio` | `0.1` | Linear warmup fraction of total steps |

### Run Benchmark

```bash
python benchmark.py \
    --model_id Qwen/Qwen2.5-Math-1.5B-Instruct \
    --test_dataset data/MATH/test.jsonl \
    --prm_path_15 checkpoints/PRM_1.5B_Train \
    --prm_path_7 checkpoints/PRM_1p5B_7B_Train \
    --rollouts 8 \
    --beam_M 4
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | Generator LLM |
| `--test_dataset` | `data/MATH/test.jsonl` | Path to MATH test set |
| `--prm_path_15` | `checkpoints/PRM_1.5B_Train` | PRM trained on 1.5B-generated data |
| `--prm_path_7` | `checkpoints/PRM_1p5B_7B_Train` | PRM trained on 7B-generated data |
| `--rollouts` | `8` | Number of rollouts $N$ per problem |
| `--beam_M` | `4` | Step proposals per beam in beam search |
| `--sampling_temperature` | `0.7` | Sampling temperature for generation |
| `--max_tokens` | `2048` | Maximum tokens per rollout |

---

## 🗂️ Repository Structure

```
.
├── benchmark.py              # Evaluate all inference methods on MATH test set
├── inference.py              # Majority vote, best-of-N variants, beam search
├── train_PRM.py              # PRM training loop
├── generate_PRM_data.py      # MC rollout data generation for PRM training
├── PRM_model.py              # PRM model definition (frozen LM + linear head)
├── Qwen-zeroShot.py          # Zero-shot baseline evaluation
├── grading/
│   ├── grader.py             # SymPy-based answer grader
│   └── math_normalize.py     # LaTeX / answer normalisation utilities
├── checkpoints/
│   ├── PRM_1.5B_Train/       # PRM trained on 1.5B-generated data
│   └── PRM_1p5B_7B_Train/    # PRM (1.5B base) trained on 7B-generated data
├── data/
│   ├── MATH/                 # MATH train / test splits
│   └── PRM_Train/            # Generated PRM training datasets
└── logs/
    ├── benchmark/            # Benchmark results per rollout count
    └── sweep/                # PRM hyperparameter sweep results
```

---

## 📚 References

[1] Snell et al. *Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters.* 2024. https://arxiv.org/abs/2408.03314

[2] Lightman et al. *Let's Verify Step by Step.* ICLR 2024. https://arxiv.org/abs/2305.20050

[3] Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. https://arxiv.org/abs/2203.11171

[4] Luo et al. *Improve Mathematical Reasoning in Language Models by Automated Process Supervision.* 2024. https://arxiv.org/abs/2406.06592

[5] Wang et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL 2024. https://arxiv.org/abs/2312.08935

[6] Guo et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. https://arxiv.org/abs/2501.12948

[7] Brown et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. https://arxiv.org/abs/2407.21787

[8] Hendrycks et al. *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS 2021. https://arxiv.org/abs/2103.03874

[9] Qwen Team. *Qwen2.5-Math Technical Report.* 2024. https://arxiv.org/abs/2412.15115

---

<sub>MIT License · Built with PyTorch & vLLM</sub>
