"""Evaluation pipeline for Variant B across all benchmarks and explanation strategies."""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import VariantBConfig
from models.common import load_tokenizer
from data.preprocess import (
    load_esnli_split,
    NLIWithExplanationDataset,
    ESNLIExplanationDataset,
    load_snli_test,
    load_multinli,
    load_anli,
)
from data.tfidf_retrieval import (
    build_tfidf_index,
    save_tfidf_index,
    load_tfidf_index,
    retrieve_explanations,
)
from data.t5_explanation import load_t5_model, generate_explanations


def evaluate_dataset(model, dataset, batch_size=64, device="cuda"):
    """Compute accuracy on a dataset."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def load_trained_model(model_dir: str, device="cuda"):
    """Load a trained Variant B model (AutoModelForSequenceClassification)."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation with gold explanations (e-SNLI test only)
# ---------------------------------------------------------------------------

def evaluate_gold(model, tokenizer, config, device="cuda"):
    """Evaluate using gold e-SNLI explanations (upper-bound performance)."""
    print("Evaluating with gold explanations on e-SNLI test...")
    test_df = load_esnli_split(config, "test")
    dataset = ESNLIExplanationDataset(test_df, tokenizer, config)
    acc = evaluate_dataset(model, dataset, config.per_device_eval_batch_size, device)
    print(f"  e-SNLI test (gold explanation): {acc:.4f}")
    return {"esnli_test_gold": acc}


# ---------------------------------------------------------------------------
# Evaluation with TF-IDF retrieved explanations
# ---------------------------------------------------------------------------

def evaluate_tfidf(model, tokenizer, config, device="cuda", tfidf_dir=None):
    """Evaluate using TF-IDF retrieved explanations across all benchmarks."""
    results = {}

    # Build or load TF-IDF index
    if tfidf_dir and Path(tfidf_dir).exists():
        print("Loading existing TF-IDF index...")
        vectorizer, train_matrix, train_explanations = load_tfidf_index(tfidf_dir)
    else:
        vectorizer, train_matrix, train_explanations = build_tfidf_index(config)
        if tfidf_dir:
            save_tfidf_index(vectorizer, train_matrix, train_explanations, tfidf_dir)

    max_len = config.max_seq_length
    batch_size = config.per_device_eval_batch_size

    # e-SNLI test
    print("TF-IDF eval on e-SNLI test...")
    test_df = load_esnli_split(config, "test")
    premises = test_df["Sentence1"].tolist()
    hypotheses = test_df["Sentence2"].tolist()
    labels = [config.label2id[lbl] for lbl in test_df["gold_label"].tolist()]
    explanations = retrieve_explanations(premises, hypotheses, vectorizer, train_matrix, train_explanations)
    dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
    results["esnli_test_tfidf"] = evaluate_dataset(model, dataset, batch_size, device)
    print(f"  e-SNLI test (TF-IDF): {results['esnli_test_tfidf']:.4f}")

    # SNLI test
    print("TF-IDF eval on SNLI test...")
    from datasets import load_dataset
    snli = load_dataset("stanfordnlp/snli", split="test")
    snli = snli.filter(lambda x: x["label"] != -1)
    premises = snli["premise"]
    hypotheses = snli["hypothesis"]
    labels = snli["label"]
    explanations = retrieve_explanations(premises, hypotheses, vectorizer, train_matrix, train_explanations)
    dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
    results["snli_test_tfidf"] = evaluate_dataset(model, dataset, batch_size, device)
    print(f"  SNLI test (TF-IDF): {results['snli_test_tfidf']:.4f}")

    # MultiNLI
    for split_name in ("validation_matched", "validation_mismatched"):
        print(f"TF-IDF eval on MultiNLI {split_name}...")
        mnli = load_dataset("nyu-mll/multi_nli", split=split_name)
        premises = mnli["premise"]
        hypotheses = mnli["hypothesis"]
        labels = mnli["label"]
        explanations = retrieve_explanations(premises, hypotheses, vectorizer, train_matrix, train_explanations)
        key = f"multinli_{split_name.replace('validation_', '')}_tfidf"
        dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
        results[key] = evaluate_dataset(model, dataset, batch_size, device)
        print(f"  MultiNLI {split_name} (TF-IDF): {results[key]:.4f}")

    # ANLI
    for r in ("r1", "r2", "r3"):
        print(f"TF-IDF eval on ANLI {r}...")
        anli = load_dataset("facebook/anli", split=f"test_{r}")
        premises = anli["premise"]
        hypotheses = anli["hypothesis"]
        labels = anli["label"]
        explanations = retrieve_explanations(premises, hypotheses, vectorizer, train_matrix, train_explanations)
        dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
        results[f"anli_{r}_tfidf"] = evaluate_dataset(model, dataset, batch_size, device)
        print(f"  ANLI {r} (TF-IDF): {results[f'anli_{r}_tfidf']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Evaluation with T5-small generated explanations
# ---------------------------------------------------------------------------

def evaluate_t5(model, tokenizer, config, t5_model_dir, device="cuda"):
    """Evaluate using T5-small generated explanations across all benchmarks."""
    results = {}

    print("Loading fine-tuned T5-small model...")
    t5_model, t5_tokenizer = load_t5_model(t5_model_dir, device)

    max_len = config.max_seq_length
    batch_size = config.per_device_eval_batch_size

    # e-SNLI test
    print("T5 eval on e-SNLI test...")
    test_df = load_esnli_split(config, "test")
    premises = test_df["Sentence1"].tolist()
    hypotheses = test_df["Sentence2"].tolist()
    labels = [config.label2id[lbl] for lbl in test_df["gold_label"].tolist()]
    explanations = generate_explanations(t5_model, t5_tokenizer, premises, hypotheses, device=device)
    dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
    results["esnli_test_t5"] = evaluate_dataset(model, dataset, batch_size, device)
    print(f"  e-SNLI test (T5): {results['esnli_test_t5']:.4f}")

    # SNLI test
    print("T5 eval on SNLI test...")
    from datasets import load_dataset
    snli = load_dataset("stanfordnlp/snli", split="test")
    snli = snli.filter(lambda x: x["label"] != -1)
    premises = snli["premise"]
    hypotheses = snli["hypothesis"]
    labels = snli["label"]
    explanations = generate_explanations(t5_model, t5_tokenizer, premises, hypotheses, device=device)
    dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
    results["snli_test_t5"] = evaluate_dataset(model, dataset, batch_size, device)
    print(f"  SNLI test (T5): {results['snli_test_t5']:.4f}")

    # MultiNLI
    for split_name in ("validation_matched", "validation_mismatched"):
        print(f"T5 eval on MultiNLI {split_name}...")
        mnli = load_dataset("nyu-mll/multi_nli", split=split_name)
        premises = mnli["premise"]
        hypotheses = mnli["hypothesis"]
        labels = mnli["label"]
        explanations = generate_explanations(t5_model, t5_tokenizer, premises, hypotheses, device=device)
        key = f"multinli_{split_name.replace('validation_', '')}_t5"
        dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
        results[key] = evaluate_dataset(model, dataset, batch_size, device)
        print(f"  MultiNLI {split_name} (T5): {results[key]:.4f}")

    # ANLI
    for r in ("r1", "r2", "r3"):
        print(f"T5 eval on ANLI {r}...")
        anli = load_dataset("facebook/anli", split=f"test_{r}")
        premises = anli["premise"]
        hypotheses = anli["hypothesis"]
        labels = anli["label"]
        explanations = generate_explanations(t5_model, t5_tokenizer, premises, hypotheses, device=device)
        dataset = NLIWithExplanationDataset(premises, hypotheses, explanations, labels, tokenizer, max_len)
        results[f"anli_{r}_t5"] = evaluate_dataset(model, dataset, batch_size, device)
        print(f"  ANLI {r} (T5): {results[f'anli_{r}_t5']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Full evaluation (all strategies)
# ---------------------------------------------------------------------------

def run_full_evaluation(model_dir: str, config: VariantBConfig = None, device="cuda",
                        tfidf_dir=None, t5_model_dir=None):
    """Run Variant B evaluation with all available explanation strategies."""
    if config is None:
        config = VariantBConfig()

    tokenizer = load_tokenizer(config.model_name)
    model = load_trained_model(model_dir, device)

    all_results = {}

    # Gold explanations (e-SNLI test only)
    all_results.update(evaluate_gold(model, tokenizer, config, device))

    # TF-IDF retrieval (all benchmarks)
    tfidf_save_dir = tfidf_dir or str(config.get_path("output_dir") / "tfidf_index")
    all_results.update(evaluate_tfidf(model, tokenizer, config, device, tfidf_save_dir))

    # T5-small generation (all benchmarks)
    if t5_model_dir is None:
        t5_model_dir = str(config.get_path("t5_output_dir"))
    if Path(t5_model_dir).exists():
        all_results.update(evaluate_t5(model, tokenizer, config, t5_model_dir, device))
    else:
        print(f"T5 model not found at {t5_model_dir}, skipping T5 evaluation.")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Variant B model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to saved Variant B model directory")
    parser.add_argument("--t5-model-dir", type=str, default=None, help="Path to fine-tuned T5-small directory")
    parser.add_argument("--tfidf-dir", type=str, default=None, help="Path to saved TF-IDF index directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/variant_b/eval_results.json")
    args = parser.parse_args()

    results = run_full_evaluation(
        model_dir=args.model_dir,
        device=args.device,
        tfidf_dir=args.tfidf_dir,
        t5_model_dir=args.t5_model_dir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(json.dumps(results, indent=2))
