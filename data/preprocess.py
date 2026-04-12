"""Dataset loading, tokenization, and MLM masking for Variant C."""

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from training.config import VariantCConfig


VALID_LABELS = {"entailment", "neutral", "contradiction"}


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def load_esnli_train(config: VariantCConfig) -> pd.DataFrame:
    """Load and concatenate the two e-SNLI training CSV files."""
    dfs = []
    for attr in ("esnli_train_1", "esnli_train_2"):
        path = config.get_path(attr)
        df = pd.read_csv(path, usecols=["gold_label", "Sentence1", "Sentence2", "Explanation_1"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["gold_label"].isin(VALID_LABELS)].dropna(subset=["Sentence1", "Sentence2", "Explanation_1"])
    df = df.reset_index(drop=True)
    return df


def load_esnli_split(config: VariantCConfig, split: str) -> pd.DataFrame:
    """Load e-SNLI dev or test split."""
    path = config.get_path(f"esnli_{split}")
    df = pd.read_csv(path, usecols=["gold_label", "Sentence1", "Sentence2", "Explanation_1"])
    df = df[df["gold_label"].isin(VALID_LABELS)].dropna(subset=["Sentence1", "Sentence2", "Explanation_1"])
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Multi-task dataset (e-SNLI with explanation MLM)
# ---------------------------------------------------------------------------

class ESNLIMultiTaskDataset(Dataset):
    """E-SNLI dataset with dynamic MLM masking on explanation tokens.

    Input format: [CLS] premise hypothesis [SEP] explanation [SEP]
    - Segment 0 (sequence_id=0): premise + hypothesis
    - Segment 1 (sequence_id=1): explanation
    MLM masking is applied only to segment 1 tokens.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, config: VariantCConfig, is_train: bool = True, seed: int = 42):
        self.premises = df["Sentence1"].tolist()
        self.hypotheses = df["Sentence2"].tolist()
        self.explanations = df["Explanation_1"].tolist()
        self.labels = [config.label2id[lbl] for lbl in df["gold_label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.mlm_probability = config.mlm_probability
        self.is_train = is_train
        self.rng = random.Random(seed)
        self.vocab_size = tokenizer.vocab_size
        self.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_a = self.premises[idx] + " " + self.hypotheses[idx]
        text_b = self.explanations[idx]

        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)        # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)

        # Identify explanation token positions via sequence_ids
        seq_ids = encoding.sequence_ids(0)
        explanation_positions = [
            i for i, sid in enumerate(seq_ids) if sid == 1
        ]

        # Build MLM labels: -100 everywhere, real token ids at masked positions
        mlm_labels = torch.full_like(input_ids, -100)

        if self.is_train and explanation_positions:
            input_ids, mlm_labels = self._apply_mlm_masking(
                input_ids, mlm_labels, explanation_positions
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "mlm_labels": mlm_labels,
        }

    def _apply_mlm_masking(self, input_ids, mlm_labels, positions):
        """Apply 15% masking to explanation positions (80% mask, 10% random, 10% keep)."""
        input_ids = input_ids.clone()
        mlm_labels = mlm_labels.clone()

        n_mask = max(1, int(len(positions) * self.mlm_probability))
        masked_positions = self.rng.sample(positions, min(n_mask, len(positions)))

        for pos in masked_positions:
            mlm_labels[pos] = input_ids[pos].item()
            r = self.rng.random()
            if r < 0.8:
                input_ids[pos] = self.mask_token_id
            elif r < 0.9:
                input_ids[pos] = self.rng.randint(0, self.vocab_size - 1)
            # else: keep original (10%)

        return input_ids, mlm_labels


# ---------------------------------------------------------------------------
# Plain NLI dataset (no explanations — for cross-domain evaluation)
# ---------------------------------------------------------------------------

class NLIDataset(Dataset):
    """Simple NLI dataset for evaluation on datasets without explanations."""

    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=384):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# External dataset loaders (SNLI, MultiNLI, ANLI)
# ---------------------------------------------------------------------------

def load_snli_test(tokenizer, max_length=384):
    """Load SNLI test set, filtering out unlabeled examples (label == -1)."""
    ds = load_dataset("stanfordnlp/snli", split="test")
    ds = ds.filter(lambda x: x["label"] != -1)
    return NLIDataset(
        premises=ds["premise"],
        hypotheses=ds["hypothesis"],
        labels=ds["label"],
        tokenizer=tokenizer,
        max_length=max_length,
    )


def load_multinli(tokenizer, split="validation_matched", max_length=384):
    """Load MultiNLI validation split (matched or mismatched)."""
    ds = load_dataset("nyu-mll/multi_nli", split=split)
    return NLIDataset(
        premises=ds["premise"],
        hypotheses=ds["hypothesis"],
        labels=ds["label"],
        tokenizer=tokenizer,
        max_length=max_length,
    )


def load_anli(tokenizer, round_tag="r1", split="test", max_length=384):
    """Load ANLI test set for a given round (r1, r2, r3)."""
    split_name = f"{split}_{round_tag}"
    ds = load_dataset("facebook/anli", split=split_name)
    return NLIDataset(
        premises=ds["premise"],
        hypotheses=ds["hypothesis"],
        labels=ds["label"],
        tokenizer=tokenizer,
        max_length=max_length,
    )
