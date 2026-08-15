"""Tests for raw Anima/Cosmos checkpoint-key conversion."""

import torch

from diffusers_anima.models.transformers.modeling_anima_transformer import (
    _convert_anima_state_dict_to_diffusers,
)


def test_cosmos_pos_embedder_helper_buffers_are_ignored() -> None:
    state = {
        "pos_embedder.dim_spatial_range": torch.zeros(1),
        "pos_embedder.dim_temporal_range": torch.zeros(1),
        "pos_embedder.seq": torch.zeros(1),
        "blocks.39.self_attn.q_proj.weight": torch.zeros(1),
        "llm_adapter.embed.weight": torch.zeros(1),
    }

    core, adapter = _convert_anima_state_dict_to_diffusers(state)

    assert set(core) == {"core.transformer_blocks.39.attn1.to_q.weight"}
    assert set(adapter) == {"llm_adapter.embed.weight"}


def test_unknown_root_key_still_fails_loudly() -> None:
    state = {"unknown.weight": torch.zeros(1)}

    import pytest

    with pytest.raises(RuntimeError, match="Unsupported Anima checkpoint key"):
        _convert_anima_state_dict_to_diffusers(state)
