"""Shared utilities for model and tokenizer loading."""

from transformers import AutoTokenizer, DebertaV2Model

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_tokenizer(model_name: str = "microsoft/deberta-v3-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    assert tokenizer.is_fast, (
        f"Fast tokenizer required for sequence_ids(). Got slow tokenizer for {model_name}."
    )
    return tokenizer


def load_base_model(model_name: str = "microsoft/deberta-v3-base"):
    return DebertaV2Model.from_pretrained(model_name)
