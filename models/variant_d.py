"""Variant D — CoT-Augmented Hybrid.

Variant D re-uses Variant C's multi-task architecture (DeBERTa-v3-base + classification
head + auxiliary MLM head on the rationale segment). The three sub-configs
(D-Human, D-CoT, D-Blend) differ only in which text fills the rationale segment;
the architecture, loss, and inference path are identical to C.
"""

from models.variant_c import DeBERTaForMultiTask as DeBERTaForVariantD

__all__ = ["DeBERTaForVariantD"]
