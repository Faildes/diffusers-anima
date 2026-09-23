"""Model helpers for Anima."""
from .anima_architecture import (
    ANIMA_29B_BASE_TO_EXPANDED,
    ANIMA_29B_INSERTED_BLOCKS,
    ANIMA_29B_NUM_LAYERS,
    ANIMA_BASE_NUM_LAYERS,
    anima_variant_from_num_layers,
    get_anima_transformer_num_layers,
)


__all__ = [
    "ANIMA_BASE_NUM_LAYERS",
    "ANIMA_29B_NUM_LAYERS",
    "ANIMA_29B_BASE_TO_EXPANDED",
    "ANIMA_29B_INSERTED_BLOCKS",
    "get_anima_transformer_num_layers",
    "anima_variant_from_num_layers",
]
