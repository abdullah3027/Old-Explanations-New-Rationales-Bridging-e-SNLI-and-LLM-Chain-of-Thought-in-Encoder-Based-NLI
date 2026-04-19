"""Training script for Variant C with custom multi-task Trainer."""

import sys
import os
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import VariantCConfig
from models.common import load_tokenizer
from models.variant_c import DeBERTaForMultiTask
from data.preprocess import load_esnli_train, load_esnli_split, ESNLIMultiTaskDataset


# ---------------------------------------------------------------------------
# Custom Trainer for dual-loss
# ---------------------------------------------------------------------------

class MultiTaskTrainer(Trainer):
    """HuggingFace Trainer subclass that handles the joint classification + MLM loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        mlm_labels = inputs.pop("mlm_labels", None)

        outputs = model(labels=labels, mlm_labels=mlm_labels, **inputs)
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        mlm_labels = inputs.pop("mlm_labels", None)

        with torch.no_grad():
            outputs = model(labels=labels, **inputs)
            loss = outputs["loss"]
            logits = outputs["logits"]

        return (loss, logits, labels)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save model using torch.save since we use nn.Module, not PreTrainedModel."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        if hasattr(self, "_variant_c_config"):
            with open(os.path.join(output_dir, "variant_c_config.json"), "w") as f:
                json.dump(vars(self._variant_c_config), f, indent=2)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load model weights from our custom model.pt checkpoint format."""
        model = model or self.model
        model_path = os.path.join(resume_from_checkpoint, "model.pt")
        if not os.path.isfile(model_path):
            raise ValueError(f"Can't find model.pt in checkpoint at {resume_from_checkpoint}")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(config: VariantCConfig = None, resume_from_checkpoint: str = None):
    if config is None:
        config = VariantCConfig()

    # Load tokenizer and data
    tokenizer = load_tokenizer(config.model_name)

    print("Loading e-SNLI training data...")
    train_df = load_esnli_train(config)
    print(f"  Training examples: {len(train_df)}")

    print("Loading e-SNLI dev data...")
    dev_df = load_esnli_split(config, "dev")
    print(f"  Dev examples: {len(dev_df)}")

    train_dataset = ESNLIMultiTaskDataset(train_df, tokenizer, config, is_train=True)
    eval_dataset = ESNLIMultiTaskDataset(dev_df, tokenizer, config, is_train=False)

    # Initialize model
    print(f"Loading model: {config.model_name}")
    model = DeBERTaForMultiTask(
        model_name=config.model_name,
        num_labels=config.num_labels,
        alpha=config.alpha,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.get_path("checkpoint_dir"),
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
        logging_steps=100,
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer._variant_c_config = config

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_dir = str(config.get_path("output_dir"))
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    print(f"Final model saved to {final_dir}/model.pt")

    return trainer, model


if __name__ == "__main__":
    train()
