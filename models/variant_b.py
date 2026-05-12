"""Variant B — Explanation-Augmented Input Classification.

Uses AutoModelForSequenceClassification (no custom architecture needed).
Input: [CLS] premise hypothesis [SEP] explanation [SEP] -> NLI label
"""

from transformers import AutoModelForSequenceClassification


def load_variant_b_model(model_name: str = "microsoft/deberta-v3-base", num_labels: int = 3):
    """Load DeBERTa for sequence classification (standard HuggingFace model)."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
