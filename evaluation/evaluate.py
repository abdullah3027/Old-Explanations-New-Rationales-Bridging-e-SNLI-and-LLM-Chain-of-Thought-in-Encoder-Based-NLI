"""Evaluation pipeline for Variant C across all benchmarks."""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import VariantCConfig
from models.common import load_tokenizer
from models.variant_c import DeBERTaForMultiTask
from data.preprocess import (
    load_esnli_split,
    NLIDataset,
    ESNLIMultiTaskDataset,
    load_snli_test,
    load_multinli,
    load_anli,
)


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
            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def load_trained_model(checkpoint_path: str, config: VariantCConfig = None, device="cuda"):
    """Load a trained Variant C model from a checkpoint."""
    if config is None:
        config = VariantCConfig()

    model = DeBERTaForMultiTask(
        model_name=config.model_name,
        num_labels=config.num_labels,
        alpha=config.alpha,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_full_evaluation(checkpoint_path: str, config: VariantCConfig = None, device="cuda"):
    """Run evaluation across all benchmarks and return results dict."""
    if config is None:
        config = VariantCConfig()

    tokenizer = load_tokenizer(config.model_name)
    model = load_trained_model(checkpoint_path, config, device)
    max_len = config.max_seq_length
    batch_size = config.per_device_eval_batch_size

    results = {}

    # 1. e-SNLI test (premise + hypothesis only — realistic inference)
    print("Evaluating on e-SNLI test (no explanation)...")
    esnli_test_df = load_esnli_split(config, "test")
    esnli_test_nli = NLIDataset(
        premises=esnli_test_df["Sentence1"].tolist(),
        hypotheses=esnli_test_df["Sentence2"].tolist(),
        labels=[config.label2id[lbl] for lbl in esnli_test_df["gold_label"].tolist()],
        tokenizer=tokenizer,
        max_length=max_len,
    )
    results["esnli_test"] = evaluate_dataset(model, esnli_test_nli, batch_size, device)
    print(f"  e-SNLI test accuracy: {results['esnli_test']:.4f}")

    # 2. SNLI test
    print("Evaluating on SNLI test...")
    snli_test = load_snli_test(tokenizer, max_len)
    results["snli_test"] = evaluate_dataset(model, snli_test, batch_size, device)
    print(f"  SNLI test accuracy: {results['snli_test']:.4f}")

    # 3. MultiNLI matched & mismatched
    for split_name in ("validation_matched", "validation_mismatched"):
        print(f"Evaluating on MultiNLI {split_name}...")
        mnli = load_multinli(tokenizer, split=split_name, max_length=max_len)
        key = f"multinli_{split_name.replace('validation_', '')}"
        results[key] = evaluate_dataset(model, mnli, batch_size, device)
        print(f"  MultiNLI {split_name} accuracy: {results[key]:.4f}")

    # 4. ANLI R1, R2, R3
    for r in ("r1", "r2", "r3"):
        print(f"Evaluating on ANLI {r}...")
        anli = load_anli(tokenizer, round_tag=r, max_length=max_len)
        results[f"anli_{r}"] = evaluate_dataset(model, anli, batch_size, device)
        print(f"  ANLI {r} accuracy: {results[f'anli_{r}']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Variant C model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/variant_c/eval_results.json")
    args = parser.parse_args()

    results = run_full_evaluation(args.checkpoint, device=args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(json.dumps(results, indent=2))
