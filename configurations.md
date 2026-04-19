
## Settings That Must Be Identical Across Variants A, B, and C

These are non-negotiable for the comparisons to be scientifically valid. If any of these differ between variants, you can't fairly attribute performance differences to the architecture — it could just be a training difference.

### 1. Model
```
model_name = "microsoft/deberta-v3-base"
```
All variants use the same backbone. Same weights to start from, same tokenizer.

---

### 2. Label Mapping
```python
label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
num_labels = 3
```
The order matters. If Variant A maps entailment → 0 but Variant B maps it → 2, the results are incomparable. This exact mapping must be used everywhere.

---

### 3. Training Hyperparameters
```
learning_rate          = 2e-5
num_train_epochs       = 3
weight_decay           = 0.01
warmup_steps           = 3090
fp16                   = True
```
These control how the model learns. Any difference here changes the training dynamics, not just the architecture.

---

### 4. Effective Batch Size
```
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
→ effective batch size = 32
```
The effective batch size (8 × 4 = 32) must be the same across all variants. Your colleagues can achieve this with different combinations (e.g., batch=16 + accumulation=2) as long as the product is 32. But if their GPU allows it, they should match exactly for consistency.

---

### 5. Evaluation Batch Size
```
per_device_eval_batch_size = 32
```
Doesn't affect model quality but should be consistent to avoid confusion when reading logs.

---

### 6. Training Data — Same Split, Same Filtering
From `preprocess.py`:
```python
VALID_LABELS = {"entailment", "neutral", "contradiction"}
# Filters out rows where gold_label not in VALID_LABELS
# Drops rows with missing Sentence1, Sentence2, Explanation_1
```
All variants must use the same filtered version of e-SNLI train, dev, and test. Same rows in, same rows out.

---

### 7. Subsampling (from the notebook)
```python
MAX_TRAIN_SAMPLES = 100_000
random_state = 42
```
The notebook subsamples 100K examples from the full ~549K training set with `random_state=42`. **All variants must use this exact same 100K subset**, otherwise they're trained on different data. The `random_state=42` seed is what guarantees they get the same rows.

---

### 8. Evaluation Benchmarks and Splits
From `evaluate.py`, every variant must be evaluated on the same benchmarks in the same way:
- e-SNLI test — **premise + hypothesis only, no explanation** (using `NLIDataset`, not `ESNLIMultiTaskDataset`)
- SNLI test (`stanfordnlp/snli`, split=`test`, filtered for label != -1)
- MultiNLI matched (`nyu-mll/multi_nli`, split=`validation_matched`)
- MultiNLI mismatched (`nyu-mll/multi_nli`, split=`validation_mismatched`)
- ANLI R1, R2, R3 (`facebook/anli`, split=`test_r1 / test_r2 / test_r3`)

---

### 9. Evaluation Metric
```
metric_for_best_model = "accuracy"
greater_is_better     = True
```
All variants report accuracy. No variant should switch to F1 or anything else.

---

### 10. Checkpointing Strategy
```
save_strategy      = "epoch"
save_total_limit   = 3
load_best_model_at_end = True
```
Best model is selected by highest dev accuracy at the end of any epoch. All variants must use the same selection criterion, otherwise you might be comparing the best epoch of one variant against the last epoch of another.

---

### 11. Dataloader
```
dataloader_num_workers = 0
```
Colab-specific. Keep at 0 across all variants to avoid multiprocessing issues.

---

### 12. GPU Environment Settings (from the notebook)
```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
model = model.float()
```
All three of these lines must be in every variant's notebook before training starts. They are not optional — they prevent OOM crashes.

---

## Settings That Are Variant C Specific — Colleagues Don't Need These

```
alpha            = 0.7       # only C has two losses
mlm_probability  = 0.15      # only C does MLM masking
max_seq_length   = 384       # C needs this because input includes explanation
                             # A can use 256; B needs ~384 too since it also appends explanation
```

---