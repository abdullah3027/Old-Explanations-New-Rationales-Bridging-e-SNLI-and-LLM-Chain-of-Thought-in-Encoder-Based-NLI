"""Dataset loading, splitting, and tokenization for Variant D (CoT-Augmented Hybrid).

Loads the merged CoT trace CSV produced by the generate_cot_traces notebooks, filters
to the rows that have a completed CoT trace (~765 of the 2,000-row target subset), and
constructs a multi-task NLI dataset where the rationale segment is sourced from either
the human explanation, the LLM-generated CoT trace, or a per-example blend.
"""

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from training.config_d import VariantDConfig
from data.preprocess import VALID_LABELS, apply_mlm_masking


# ---------------------------------------------------------------------------
# CoT subset loading + stratified split
# ---------------------------------------------------------------------------

def load_cot_subset(config: VariantDConfig) -> pd.DataFrame:
    """Load the merged CoT traces CSV and filter to rows usable for training."""
    path = config.get_path("cot_traces_csv")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"CoT traces CSV not found at {path}. "
            f"Run the generate_cot_traces notebooks (Part 1 + Part 2) and merge the outputs first."
        )

    df = pd.read_csv(path)

    required_cols = {
        "pair_id",
        "gold_label",
        "Sentence1",
        "Sentence2",
        config.rationale_col_human,
        config.rationale_col_cot,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CoT CSV missing required columns: {sorted(missing)}")

    df = df[df["gold_label"].isin(VALID_LABELS)]
    df = df.dropna(subset=["Sentence1", "Sentence2", config.rationale_col_human, config.rationale_col_cot])
    df = df[
        (df[config.rationale_col_human].astype(str).str.strip() != "")
        & (df[config.rationale_col_cot].astype(str).str.strip() != "")
    ]
    df = df.sort_values("pair_id").reset_index(drop=True)
    return df


def split_cot_subset(df: pd.DataFrame, dev_frac: float = 0.1, seed: int = 42):
    """Stratified 90/10 split by gold_label so the dev set stays balanced."""
    train_pieces, dev_pieces = [], []
    for label in sorted(df["gold_label"].unique()):
        group = df[df["gold_label"] == label].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_dev = max(1, int(round(len(group) * dev_frac)))
        dev_pieces.append(group.iloc[:n_dev])
        train_pieces.append(group.iloc[n_dev:])

    train_df = pd.concat(train_pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev_df = pd.concat(dev_pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, dev_df


# ---------------------------------------------------------------------------
# Variant D multi-task dataset
# ---------------------------------------------------------------------------

class ESNLIVariantDDataset(Dataset):
    """E-SNLI dataset for Variant D with dynamic MLM masking on the rationale segment.

    Input format: [CLS] premise hypothesis [SEP] rationale [SEP]
      - Segment 0 (sequence_id=0): premise + " " + hypothesis
      - Segment 1 (sequence_id=1): rationale text chosen by sub_config

    sub_config:
      "human" -> rationale = Explanation_1
      "cot"   -> rationale = cot_rationale_esnli_style
      "blend" -> per-row deterministic sampling using random.Random(blend_seed + idx);
                 with probability blend_ratio use the human explanation, else the CoT rationale.

    MLM masking is applied only to segment-1 (rationale) tokens.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, config: VariantDConfig, is_train: bool = True, seed: int = 42):
        self.premises = df["Sentence1"].tolist()
        self.hypotheses = df["Sentence2"].tolist()
        self.human_rationales = df[config.rationale_col_human].tolist()
        self.cot_rationales = df[config.rationale_col_cot].tolist()
        self.labels = [config.label2id[lbl] for lbl in df["gold_label"].tolist()]

        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.mlm_probability = config.mlm_probability
        self.is_train = is_train

        self.sub_config = config.sub_config
        self.blend_ratio = config.blend_ratio
        self.blend_seed = config.blend_seed

        self.rng = random.Random(seed)
        self.vocab_size = tokenizer.vocab_size
        self.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    def __len__(self):
        return len(self.labels)

    def _select_rationale(self, idx: int) -> str:
        if self.sub_config == "human":
            return self.human_rationales[idx]
        if self.sub_config == "cot":
            return self.cot_rationales[idx]
        # blend: deterministic per-row choice so reruns with the same blend_seed
        # produce bit-for-bit identical training sets.
        per_row_rng = random.Random(self.blend_seed + idx)
        return self.human_rationales[idx] if per_row_rng.random() < self.blend_ratio else self.cot_rationales[idx]

    def __getitem__(self, idx):
        text_a = self.premises[idx] + " " + self.hypotheses[idx]
        text_b = self._select_rationale(idx)

        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        seq_ids = encoding.sequence_ids(0)
        rationale_positions = [i for i, sid in enumerate(seq_ids) if sid == 1]

        mlm_labels = torch.full_like(input_ids, -100)

        if self.is_train and rationale_positions:
            input_ids, mlm_labels = apply_mlm_masking(
                input_ids,
                mlm_labels,
                rationale_positions,
                self.mlm_probability,
                self.mask_token_id,
                self.vocab_size,
                self.rng,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "mlm_labels": mlm_labels,
        }
