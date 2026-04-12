"""Hyperparameter configuration for Variant C (Multi-Task Learning)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VariantCConfig:
    # Model
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 3

    # Sequence lengths
    max_seq_length: int = 384

    # Training
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8      # reduced from 16 to fit T4 16GB with seq_len=384
    per_device_eval_batch_size: int = 32      # reduced from 64 to be safe during eval
    gradient_accumulation_steps: int = 4      # increased from 2; effective batch = 8*4 = 32
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 3090                  # 6% of ~51500 total steps (replaces warmup_ratio)
    fp16: bool = True

    # Multi-task loss weighting
    alpha: float = 0.7  # weight for classification loss; (1-alpha) for MLM loss
    mlm_probability: float = 0.15

    # Label mapping
    label2id: dict = field(default_factory=lambda: {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    })
    id2label: dict = field(default_factory=lambda: {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    })

    # Paths (relative to project root)
    project_root: str = ""
    esnli_train_1: str = "Datasets/E-SNLI/esnli_train_1.csv"
    esnli_train_2: str = "Datasets/E-SNLI/esnli_train_2.csv"
    esnli_dev: str = "Datasets/E-SNLI/esnli_dev.csv"
    esnli_test: str = "Datasets/E-SNLI/esnli_test.csv"
    output_dir: str = "results/variant_c"
    checkpoint_dir: str = "results/variant_c/checkpoints"

    # Dataloader
    dataloader_num_workers: int = 2

    # Checkpointing
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"

    def get_path(self, attr: str) -> Path:
        root = Path(self.project_root) if self.project_root else Path(".")
        return root / getattr(self, attr)
