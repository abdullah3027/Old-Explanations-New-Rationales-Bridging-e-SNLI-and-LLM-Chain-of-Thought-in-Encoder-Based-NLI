"""Training script for Variant D — runs one sub-config (human / cot / blend) per call.

Reuses MultiTaskTrainer and compute_metrics from training.train so the dual-loss
behavior, prediction step, and custom checkpoint format match Variant C exactly.
"""

import sys
import os
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config_d import VariantDConfig, VALID_SUB_CONFIGS
from training.train import MultiTaskTrainer, compute_metrics
from models.common import load_tokenizer
from models.variant_d import DeBERTaForVariantD
from data.preprocess_d import load_cot_subset, split_cot_subset, ESNLIVariantDDataset


def train(config: VariantDConfig = None, resume_from_checkpoint: str = None):
    if config is None:
        config = VariantDConfig()

    tokenizer = load_tokenizer(config.model_name)

    print(f"Loading CoT subset from {config.get_path('cot_traces_csv')} ...")
    full_df = load_cot_subset(config)
    print(f"  Total CoT rows after filtering: {len(full_df)}")

    train_df, dev_df = split_cot_subset(full_df, dev_frac=config.dev_frac, seed=config.split_seed)
    print(f"  Train: {len(train_df)}  |  Dev: {len(dev_df)}")
    print(f"  Sub-config: {config.sub_config}  |  blend_ratio: {config.blend_ratio}")

    train_dataset = ESNLIVariantDDataset(train_df, tokenizer, config, is_train=True)
    eval_dataset = ESNLIVariantDDataset(dev_df, tokenizer, config, is_train=False)

    print(f"Loading model: {config.model_name}")
    model = DeBERTaForVariantD(
        model_name=config.model_name,
        num_labels=config.num_labels,
        alpha=config.alpha,
    )

    checkpoint_dir = str(config.get_path("checkpoint_dir"))
    output_dir = str(config.get_path("output_dir"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        eval_strategy="epoch",
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        dataloader_num_workers=config.dataloader_num_workers,
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # Note: deliberately *not* setting trainer._variant_c_config here. That attribute makes
    # MultiTaskTrainer.save_model emit a "variant_c_config.json" file next to every checkpoint,
    # which would be a misleading name for D. We write variant_d_config.json explicitly below.

    print(f"Starting training for sub_config={config.sub_config} ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    final_config_path = os.path.join(output_dir, "variant_d_config.json")
    with open(final_config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    print(f"Final config saved to {final_config_path}")

    return trainer, model


def train_all_sub_configs(base_config_kwargs: dict = None):
    """Train D-Human, D-CoT, and D-Blend back-to-back. Returns dict of trained models."""
    base_kwargs = dict(base_config_kwargs or {})
    results = {}
    for sc in ("human", "cot", "blend"):
        kwargs = dict(base_kwargs)
        kwargs["sub_config"] = sc
        config = VariantDConfig(**kwargs)
        print("=" * 70)
        print(f"Training Variant D sub-config: {sc}")
        print("=" * 70)
        trainer, model = train(config=config)
        results[sc] = {"trainer": trainer, "model": model, "config": config}

        # Free GPU memory between runs
        del trainer, model
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Variant D.")
    parser.add_argument(
        "--sub-config",
        choices=sorted(VALID_SUB_CONFIGS) + ["all"],
        default="human",
        help="Which D sub-config to train; 'all' runs human, cot, blend back-to-back.",
    )
    parser.add_argument("--blend-ratio", type=float, default=0.5)
    args = parser.parse_args()

    if args.sub_config == "all":
        train_all_sub_configs({"blend_ratio": args.blend_ratio})
    else:
        cfg = VariantDConfig(sub_config=args.sub_config, blend_ratio=args.blend_ratio)
        train(config=cfg)
