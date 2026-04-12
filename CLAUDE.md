# Old Explanations, New Rationales: Bridging e-SNLI and LLM Chain-of-Thought in Encoder-Based NLI

## Project Overview

This project investigates whether integrating natural language reasoning into the training process of a transformer encoder improves NLI classification accuracy, and which architectural strategy is most effective. It compares four DeBERTa-v3-based architectural variants on the e-SNLI dataset, with a novel fourth variant that blends 2018-era human explanations with modern LLM-generated Chain-of-Thought rationales.

**Key Research Question:** Does incorporating reasoning explanations into encoder training improve NLI accuracy, and what is the best way to do it?

**Key Paper:** "e-SNLI: Natural Language Inference with Natural Language Explanations" (Camburu et al., NeurIPS 2018) — https://arxiv.org/abs/1812.01193

**Team Size:** 3 people
**Timeline:** ~20 working days (8-week course project)
**Compute:** Google Colab T4 GPU, 2–3 hour sessions max

---

## Architectural Variants

### Variant A — Classification Only (Baseline)
- **Input:** Premise + Hypothesis
- **Training Objective:** Cross-entropy on NLI label
- **Details:** DeBERTa-v3-base takes premise and hypothesis, predicts label directly. No explanation involved. Standard NLI approach. Serves as primary baseline.
- **T4 Train Time:** ~1.5 hours
- **Owner:** P2

### Variant B — Explanation-Augmented Input
- **Input:** Premise + Hypothesis + Explanation (concatenated)
- **Training Objective:** Cross-entropy on NLI label
- **Details:** DeBERTa-v3-base takes all three as input, predicts label. At test time (no gold explanation available), two strategies are compared:
  1. Retrieve the most similar training example's explanation using TF-IDF cosine similarity
  2. Generate an explanation using T5-small (60M parameters, runs on CPU in seconds)
- **Tests:** Whether providing reasoning as additional context helps the classifier
- **T4 Train Time:** ~2 hours
- **Owner:** P2

### Variant C — Multi-Task Learning (Classify + Explain)
- **Input:** Premise + Hypothesis + Explanation
- **Training Objective:** Label cross-entropy + explanation masked language modeling (joint loss)
- **Details:** DeBERTa-v3-base trained with two objectives simultaneously: (1) predict NLI label via classification head, (2) predict masked tokens within the explanation via secondary MLM head. Forces the encoder to build representations that capture reasoning patterns. Explanation head discarded during inference.
- **Tests:** Whether learning to explain improves classification
- **T4 Train Time:** ~2.5 hours
- **Owner:** P3

### Variant D — CoT-Augmented Hybrid (Novel Contribution)
Added per professor feedback. Bridges e-SNLI human explanations with modern LLM-generated CoT rationales. Three sub-configurations:

- **D-Human:** Trained using only e-SNLI human explanations (same data as B/C but with Variant D's architecture)
- **D-CoT:** Trained using only LLM-generated CoT rationales
- **D-Blend:** Trained on a mixture of human explanations and CoT rationales (blend ratio is an ablation axis)

**CoT Generation Pipeline:** Use a free LLM API (e.g., HuggingFace Inference API with DeepSeek-R1-Distill models) or a local smaller model to generate multi-step reasoning traces for a subset of training examples. CoT traces are 3–5 sentences with explicit logical steps.

- **Owner:** P1 (CoT generation pipeline + D sub-configs)

---

## Models

| Model | Parameters | Purpose | HuggingFace ID |
|-------|-----------|---------|----------------|
| DeBERTa-v3-base | 184M | Main model for all variants (A, B, C, D) | `microsoft/deberta-v3-base` |
| DeBERTa-v3-large | 304M | Scale comparison experiment | `microsoft/deberta-v3-large` |
| T5-small | 60M | Generate explanations at test time for Variant B | `google-t5/t5-small` |

All models fit comfortably on a T4 GPU (16GB VRAM) with batch size 16–32. Each training run completes within 1.5–2.5 hours on 3 epochs.

---

## Datasets

| Dataset | Size | Purpose | HuggingFace ID |
|---------|------|---------|----------------|
| e-SNLI | 570K train / 10K test | Training (all variants) — includes human explanations | `esnli` |
| SNLI | 570K train / 10K test | Baseline training (labels only) | `stanfordnlp/snli` |
| MultiNLI | 433K (10K matched + 10K mismatched dev) | Out-of-domain generalization test | `nyu-mll/multi_nli` |
| ANLI | R1/R2/R3 test sets (1K/1K/1.2K) | Adversarial hard test | `facebook/anli` |

No data generation or collection needed for Variants A–C. Variant D requires generating CoT traces for a subset.

---

## Evaluation Plan

### Metrics
1. **Classification Accuracy:** Percentage of correctly predicted NLI labels on each test set (e-SNLI test, MultiNLI matched/mismatched, ANLI R1/R2/R3)
2. **Faithfulness Rate:** Percentage of examples where perturbing the premise causes corresponding change in both prediction and behavior. Tests whether models are reasoning vs. memorizing.

### Ablation Studies
1. **Architecture ablation:** Variant A vs. B vs. C vs. D (and D sub-configs)
2. **Data source ablation (Variant D):** D-Human vs. D-CoT vs. D-Blend
3. **Blend ratio ablation:** What mix of human/CoT explanations works best?
4. **Data size ablation:** Train on 10%, 25%, 50%, 100% of training set
5. **Scale comparison:** DeBERTa-v3-base vs. DeBERTa-v3-large for baseline and best variant (does explanation training at base scale match a larger model without explanations?)
6. **Test-time explanation strategy (Variant B):** TF-IDF retrieval vs. T5-small generation

### Benchmarks
- **In-domain:** e-SNLI test set
- **Cross-domain:** MultiNLI matched + mismatched
- **Adversarial:** ANLI R1, R2, R3

---

## Innovation / Novelty

1. **Systematic architectural comparison:** No prior work has compared classification-only, explanation-as-input, multi-task explanation training, and CoT-augmented hybrid on the same NLI dataset with the same base model.
2. **Bridging old and new reasoning:** Variant D is the core novelty — comparing 2018-era human explanations with modern LLM CoT rationales and their blend in encoder-based NLI.
3. **Faithfulness-controlled evaluation:** Premise perturbation testing to distinguish genuine reasoning from pattern memorization.
4. **Practical efficiency question:** Can adding reasoning data to a small model match simply scaling model size?

---

## Team Roles (3 People)

| Person | Primary Responsibilities |
|--------|------------------------|
| **P1** | CoT generation pipeline, Variant D (D-CoT, D-Human, D-Blend), blend ratio ablations |
| **P2** | Variant A (baseline), Variant B (explanation-augmented), test-time explanation strategies (TF-IDF retrieval + T5-small), data size ablations |
| **P3** | Variant C (multi-task), evaluation pipeline, faithfulness perturbation testing, scale comparison (DeBERTa-large), error analysis |

All team members share: dataset preprocessing, infrastructure setup, report writing.

---

## 20-Day Implementation Plan

### Phase 1: Infrastructure & Data (Days 1–4)
- **Day 1–2:** Download all datasets. Set up shared Colab notebook infrastructure with checkpointing. Preprocess e-SNLI into formats for all variants.
- **Day 3–4:** P1 starts CoT generation pipeline. P2 implements Variant A training loop. P3 implements evaluation harness.

### Phase 2: Baselines & Core Variants (Days 5–10)
- **Day 5–6:** Train Variant A (P2). P1 continues CoT generation. P3 implements Variant C architecture.
- **Day 7–8:** Train Variant B with gold explanations (P2). Train Variant C (P3). P1 completes CoT generation for subset.
- **Day 9–10:** Evaluate A, B, C on e-SNLI test. **HALFWAY MILESTONE:** Accuracy numbers for A, B, C reported.

### Phase 3: Variant D & Ablations (Days 11–16)
- **Day 11–12:** Train D-Human, D-CoT, D-Blend (P1). Data size ablation runs (P2).
- **Day 13–14:** Evaluate all D sub-configs. Blend ratio ablation (P1). Test-time explanation strategy comparison for B (P2).
- **Day 15–16:** Scale comparison with DeBERTa-large (P3). Faithfulness perturbation experiments (P3).

### Phase 4: Analysis & Report (Days 17–20)
- **Day 17–18:** Full evaluation across all benchmarks (MultiNLI, ANLI). Error analysis by linguistic phenomenon (negation, quantifiers, lexical inference, world knowledge).
- **Day 19–20:** Compile results tables. Write final report. Clean code for reproducibility.

---

## Hyperparameters (Default Starting Points)

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/deberta-v3-base` |
| Learning rate | 2e-5 |
| Batch size | 16–32 (T4 dependent) |
| Epochs | 3 |
| Max sequence length | 256 (Variant A), 384 (Variants B, C, D) |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup steps | 6% of total |
| Loss weighting (Variant C) | α × CE_label + (1-α) × MLM_explanation, α=0.7 |
| Gradient accumulation | As needed to fit T4 memory |

---

## Compute Budget (Colab T4 Sessions)

| Task | Sessions | Time Each |
|------|----------|-----------|
| Variant A training | 1 | ~1.5 hrs |
| Variant B training | 1 | ~2 hrs |
| Variant C training | 1 | ~2.5 hrs |
| Variant D (3 sub-configs) | 3 | ~2 hrs each |
| DeBERTa-large baseline + best variant | 2 | ~2.5 hrs each |
| Data size ablations (4 runs) | 2 | ~2 hrs each |
| Evaluation + faithfulness testing | 1–2 | ~2 hrs each |
| **Total** | **~12–14 sessions** | |

---

## Key Links

### Datasets
- e-SNLI: https://huggingface.co/datasets/esnli
- SNLI: https://huggingface.co/datasets/stanfordnlp/snli
- MultiNLI: https://huggingface.co/datasets/nyu-mll/multi_nli
- ANLI: https://huggingface.co/datasets/facebook/anli

### Models
- DeBERTa-v3-base: https://huggingface.co/microsoft/deberta-v3-base
- DeBERTa-v3-large: https://huggingface.co/microsoft/deberta-v3-large
- T5-small: https://huggingface.co/google-t5/t5-small

### Libraries
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- BERTScore: https://github.com/Tiiiger/bert_score

### Paper
- e-SNLI (Camburu et al., NeurIPS 2018): https://arxiv.org/abs/1812.01193

---

## Framework & Tools
- **Language:** Python
- **Framework:** PyTorch
- **Training:** HuggingFace Transformers + Trainer API
- **Data loading:** HuggingFace Datasets (`load_dataset("esnli")`)
- **Compute:** Google Colab T4 GPU (free tier, 2–3 hr sessions)
- **Checkpointing:** Save after every epoch; resume across sessions

---

## Expected Outcomes

1. **Reasoning helps NLI:** At least one explanation-enhanced variant (B, C, or D) outperforms the classification-only baseline A, especially on out-of-domain (MultiNLI) and adversarial (ANLI) benchmarks.
2. **Architecture matters:** Multi-task (C) and CoT-augmented (D) expected to outperform simple concatenation (B), because they force deeper reasoning integration.
3. **Blend outperforms either source alone:** D-Blend expected to beat D-Human and D-CoT individually, showing complementary value of old explanations and new rationales.
4. **Efficiency insight:** Explanation training at base scale (184M + explanations) may approach or match DeBERTa-large (304M) without explanations — cheaper to add reasoning data than to scale model size.
5. **Faithfulness:** Explanation-enhanced models expected to show higher faithfulness rates on perturbation tests.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Colab session disconnects mid-training | Checkpoint every epoch; resume from last checkpoint |
| CoT generation takes too long | Use smaller subset (10K–20K); use free HuggingFace Inference API |
| Variant C MLM head is tricky to implement | Start with simpler auxiliary loss; fallback to explanation token prediction |
| No accuracy improvement from explanations | This is still a valid finding; document negative results; focus on faithfulness analysis |
| T4 OOM on longer sequences | Reduce max_seq_length; use gradient accumulation; reduce batch size |

---

## File Structure (Suggested)

```
project/
├── data/
│   ├── preprocess.py          # Dataset loading and formatting for all variants
│   └── cot_generation.py      # CoT trace generation pipeline (Variant D)
├── models/
│   ├── variant_a.py           # Classification-only baseline
│   ├── variant_b.py           # Explanation-augmented input
│   ├── variant_c.py           # Multi-task (classify + explain)
│   ├── variant_d.py           # CoT-augmented hybrid
│   └── common.py              # Shared model utilities, DeBERTa loading
├── evaluation/
│   ├── evaluate.py            # Accuracy evaluation across all benchmarks
│   ├── faithfulness.py        # Premise perturbation faithfulness testing
│   └── metrics.py             # BERTScore and other metric utilities
├── training/
│   ├── train.py               # Main training loop with checkpointing
│   └── config.py              # Hyperparameter configs for all variants
├── notebooks/
│   ├── train_variant_a.ipynb  # Colab notebook for Variant A
│   ├── train_variant_b.ipynb  # Colab notebook for Variant B
│   ├── train_variant_c.ipynb  # Colab notebook for Variant C
│   ├── train_variant_d.ipynb  # Colab notebook for Variant D (3 sub-configs)
│   └── evaluate_all.ipynb     # Colab notebook for full evaluation
├── results/
│   └── ...                    # Saved metrics, tables, plots
├── claude.md                  # This file — project reference
└── README.md
```
