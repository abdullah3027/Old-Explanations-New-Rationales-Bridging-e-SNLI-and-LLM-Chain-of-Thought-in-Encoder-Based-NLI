"""T5-small explanation generation for Variant B test-time inference.

Fine-tunes T5-small on (premise + hypothesis) -> explanation,
then generates explanations for test examples at inference time.
"""

import sys
import os
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import VariantBConfig
from data.preprocess import load_esnli_train


class T5ExplanationDataset(Dataset):
    """Dataset for fine-tuning T5-small on NLI explanation generation.

    Input: "nli: premise: {P} hypothesis: {H}"
    Target: explanation text
    """

    def __init__(self, premises, hypotheses, explanations, tokenizer, max_input_length=256, max_target_length=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.explanations = explanations
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        input_text = f"nli: premise: {self.premises[idx]} hypothesis: {self.hypotheses[idx]}"

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            self.explanations[idx],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


def finetune_t5(config: VariantBConfig = None):
    """Fine-tune T5-small on explanation generation using a subset of training data."""
    if config is None:
        config = VariantBConfig()

    tokenizer = AutoTokenizer.from_pretrained(config.t5_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model_name)

    print("Loading e-SNLI training data for T5 fine-tuning...")
    train_df = load_esnli_train(config)

    # Sample subset for efficiency
    if config.t5_train_subset_size < len(train_df):
        train_df = train_df.sample(n=config.t5_train_subset_size, random_state=42).reset_index(drop=True)
        print(f"  Using {len(train_df)} training examples (subset)")
    else:
        print(f"  Using all {len(train_df)} training examples")

    dataset = T5ExplanationDataset(
        premises=train_df["Sentence1"].tolist(),
        hypotheses=train_df["Sentence2"].tolist(),
        explanations=train_df["Explanation_1"].tolist(),
        tokenizer=tokenizer,
        max_input_length=config.t5_max_input_length,
        max_target_length=config.t5_max_target_length,
    )

    output_dir = str(config.get_path("t5_output_dir"))

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.t5_epochs,
        per_device_train_batch_size=config.t5_batch_size,
        per_device_eval_batch_size=config.t5_batch_size,
        learning_rate=config.t5_learning_rate,
        weight_decay=0.01,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
        predict_with_generate=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting T5-small fine-tuning...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"T5-small model saved to {output_dir}")

    return model, tokenizer


def load_t5_model(model_dir: str, device="cuda"):
    """Load a fine-tuned T5-small model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_explanations(model, tokenizer, premises, hypotheses, batch_size=32, max_length=128, device="cuda"):
    """Generate explanations for test examples using fine-tuned T5-small.

    Args:
        model: Fine-tuned T5 model
        tokenizer: T5 tokenizer
        premises: List of premise strings
        hypotheses: List of hypothesis strings
        batch_size: Generation batch size
        max_length: Maximum generated sequence length
        device: Device to run generation on

    Returns:
        List of generated explanation strings
    """
    model.eval()
    all_explanations = []

    for i in range(0, len(premises), batch_size):
        batch_premises = premises[i : i + batch_size]
        batch_hypotheses = hypotheses[i : i + batch_size]

        input_texts = [
            f"nli: premise: {p} hypothesis: {h}"
            for p, h in zip(batch_premises, batch_hypotheses)
        ]

        inputs = tokenizer(
            input_texts,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_explanations.extend(decoded)

        if (i // batch_size) % 10 == 0:
            print(f"  Generated explanations for {min(i + batch_size, len(premises))}/{len(premises)} examples")

    return all_explanations


if __name__ == "__main__":
    finetune_t5()
