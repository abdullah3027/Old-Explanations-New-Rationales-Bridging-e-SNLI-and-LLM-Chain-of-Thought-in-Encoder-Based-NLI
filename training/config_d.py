"""Hyperparameter configuration for Variant D (CoT-Augmented Hybrid).

Three sub-configs share this dataclass — only `sub_config` and `blend_ratio` differ:
  - "human" : trains with Explanation_1 in the rationale segment
  - "cot"   : trains with cot_rationale_esnli_style in the rationale segment
  - "blend" : per-example sampling between the two, governed by blend_ratio

The architecture (DeBERTa-v3-base + classification head + MLM head) and all training
invariants match Variant C; only `warmup_steps` is recomputed for the smaller dataset
(765 CoT-complete rows -> ~689 train / ~76 dev after 90/10 stratified split).
"""

from dataclasses import dataclass, field
from pathlib import Path


VALID_SUB_CONFIGS = {"human", "cot", "blend"}


@dataclass
class VariantDConfig:
    # Model
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 3

    # Sequence lengths
    max_seq_length: int = 384

    # Training (matches Variant C invariants)
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4       # effective batch = 8 * 4 = 32
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 4                      # ~6% of ~63 total optimizer steps on ~689 train rows
    fp16: bool = True

    # Multi-task loss weighting (same as Variant C)
    alpha: float = 0.7
    mlm_probability: float = 0.15

    # Variant D specific
    sub_config: str = "human"                  # one of {"human", "cot", "blend"}
    rationale_col_human: str = "Explanation_1"
    rationale_col_cot: str = "cot_rationale_esnli_style"
    blend_ratio: float = 0.5                   # P(use human) for D-Blend; ignored for human/cot
    blend_seed: int = 42

    # Label mapping (must match A/B/C)
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
    cot_traces_csv: str = "Datasets/CoT/cot_traces.csv"
    esnli_test: str = "Datasets/E-SNLI/esnli_test.csv"
    dev_frac: float = 0.1                      # ~689 train / ~76 dev split (765 CoT-complete rows)
    split_seed: int = 42

    # Dataloader
    dataloader_num_workers: int = 0

    # Checkpointing
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"

    def __post_init__(self):
        if self.sub_config not in VALID_SUB_CONFIGS:
            raise ValueError(
                f"sub_config must be one of {sorted(VALID_SUB_CONFIGS)}, got {self.sub_config!r}"
            )

    @property
    def output_dir(self) -> str:
        return f"results/variant_d_{self.sub_config}"

    @property
    def checkpoint_dir(self) -> str:
        return f"results/variant_d_{self.sub_config}/checkpoints"

    def get_path(self, attr_or_dir: str) -> Path:
        root = Path(self.project_root) if self.project_root else Path(".")
        if attr_or_dir in ("output_dir", "checkpoint_dir"):
            return root / getattr(self, attr_or_dir)
        return root / getattr(self, attr_or_dir)
