"""Architecture metadata shared by Anima loaders and adapters.

The 40-block Anima 2.9B checkpoint was created by inserting twelve blocks into
the original 28-block Anima layout.  Block numbers are therefore not stable
between the two architectures.  Keep the measured correspondence in one place
so features such as legacy LoRA loading do not silently target the wrong layer.
"""

from __future__ import annotations

from typing import Any


ANIMA_BASE_NUM_LAYERS = 28
ANIMA_29B_NUM_LAYERS = 40

# Original Anima block index -> the bit-identical inherited block in Anima 2.9B.
# The mapping was established by comparing the released base and 2.9B weights.
ANIMA_29B_BASE_TO_EXPANDED: tuple[int, ...] = (
    0,
    1,
    3,
    4,
    6,
    7,
    9,
    10,
    12,
    13,
    15,
    16,
    18,
    19,
    20,
    22,
    23,
    25,
    26,
    28,
    29,
    31,
    32,
    34,
    35,
    37,
    38,
    39,
)

ANIMA_29B_INSERTED_BLOCKS: frozenset[int] = frozenset(
    set(range(ANIMA_29B_NUM_LAYERS)) - set(ANIMA_29B_BASE_TO_EXPANDED)
)


def get_anima_transformer_num_layers(transformer: Any) -> int | None:
    """Return the active main-transformer depth without assuming one layout."""
    config = getattr(transformer, "config", None)
    configured = getattr(config, "num_layers", None)
    if configured is None and hasattr(config, "get"):
        configured = config.get("num_layers")
    if configured is not None:
        try:
            value = int(configured)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value

    core = getattr(transformer, "core", None)
    for candidate in (core, transformer):
        blocks = getattr(candidate, "transformer_blocks", None)
        if blocks is None:
            blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            try:
                value = len(blocks)
            except TypeError:
                continue
            if value > 0:
                return int(value)
    return None


def anima_variant_from_num_layers(num_layers: int | None) -> str:
    if num_layers == ANIMA_BASE_NUM_LAYERS:
        return "base"
    if num_layers == ANIMA_29B_NUM_LAYERS:
        return "2.9b"
    return "custom"
