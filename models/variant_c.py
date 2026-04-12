"""Variant C — Multi-Task DeBERTa (Classification + Explanation MLM)."""

import torch
import torch.nn as nn
from transformers import DebertaV2Model


class DeBERTaForMultiTask(nn.Module):
    """DeBERTa-v3 with dual heads: NLI classification + explanation MLM.

    Training: joint loss = alpha * CE_label + (1 - alpha) * MLM_explanation
    Inference: only the classification head is used; MLM head is discarded.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base", num_labels: int = 3, alpha: float = 0.7):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size
        vocab_size = self.deberta.config.vocab_size
        self.alpha = alpha

        # Classification head ([CLS] -> label)
        self.cls_dropout = nn.Dropout(0.1)
        self.cls_head = nn.Linear(hidden_size, num_labels)

        # MLM head (sequence -> masked token prediction)
        self.mlm_dense = nn.Linear(hidden_size, hidden_size)
        self.mlm_act = nn.GELU()
        self.mlm_norm = nn.LayerNorm(hidden_size)
        self.mlm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie MLM decoder weights to input embeddings
        self.mlm_decoder.weight = self.deberta.embeddings.word_embeddings.weight

        self._init_heads()

    def _init_heads(self):
        """Initialize classification and MLM head weights."""
        for module in [self.cls_head, self.mlm_dense]:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None, mlm_labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        # Classification logits from [CLS] token
        cls_logits = self.cls_head(self.cls_dropout(sequence_output[:, 0]))

        loss = None
        cls_loss = None
        mlm_loss = None

        if labels is not None:
            cls_loss = nn.CrossEntropyLoss()(cls_logits, labels)
            loss = self.alpha * cls_loss

            if mlm_labels is not None:
                mlm_hidden = self.mlm_decoder(self.mlm_norm(self.mlm_act(self.mlm_dense(sequence_output))))
                mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    mlm_hidden.view(-1, mlm_hidden.size(-1)),
                    mlm_labels.view(-1),
                )
                loss = loss + (1 - self.alpha) * mlm_loss

        return {
            "loss": loss,
            "logits": cls_logits,
            "cls_loss": cls_loss.item() if cls_loss is not None else None,
            "mlm_loss": mlm_loss.item() if mlm_loss is not None else None,
        }
