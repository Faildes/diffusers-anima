"""Tests for state-dict wrapping-prefix handling used by ``from_single_file``."""

from __future__ import annotations

import torch

from diffusers_anima.pipelines.anima.loading import (
    _extract_qwen35_causal_state_dict,
    _strip_wrapping_prefixes,
    infer_anima_num_layers,
)


def _sample_anima_like_state_dict() -> dict[str, torch.Tensor]:
    """Keys representative of a real Anima transformer state dict (root + blocks + adapter)."""
    return {
        "x_embedder.proj.1.weight": torch.zeros(1),
        "t_embedder.1.linear_1.weight": torch.zeros(1),
        "blocks.0.self_attn.q_proj.weight": torch.zeros(1),
        "blocks.27.mlp.layer2.weight": torch.zeros(1),
        "llm_adapter.embed.weight": torch.zeros(1),
    }


def _prefix_keys(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def test_strip_wrapping_prefixes_passes_through_unprefixed_state_dict() -> None:
    base = _sample_anima_like_state_dict()

    result = _strip_wrapping_prefixes(base)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_net_wrapper() -> None:
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "net.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_model_wrapper() -> None:
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "model.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_diffusion_model_wrapper() -> None:
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "diffusion_model.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_composite_comfyui_wrapper() -> None:
    """waiANIMA_v10 ships keys like ``model.diffusion_model.x_embedder.proj.1.weight``."""
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "model.diffusion_model.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_preserves_tensors() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    wrapped = {"model.diffusion_model.x_embedder.proj.1.weight": tensor}

    result = _strip_wrapping_prefixes(wrapped)

    assert "x_embedder.proj.1.weight" in result
    assert torch.equal(result["x_embedder.proj.1.weight"], tensor)


def test_strip_wrapping_prefixes_does_not_strip_when_prefix_is_not_shared() -> None:
    """If only some keys start with a prefix, leave all keys alone to avoid silent merges."""
    mixed = {
        "model.x_embedder.proj.1.weight": torch.zeros(1),
        "x_embedder.proj.1.weight": torch.zeros(1),  # same key would result after stripping
    }

    result = _strip_wrapping_prefixes(mixed)

    assert set(result.keys()) == set(mixed.keys())


def test_strip_wrapping_prefixes_on_empty_state_dict() -> None:
    assert _strip_wrapping_prefixes({}) == {}


def _synthetic_block_state_dict(num_layers: int, prefix: str = "") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}blocks.{index}.self_attn.q_proj.weight": torch.zeros(1)
        for index in range(num_layers)
    }


def test_infer_anima_num_layers_original_28() -> None:
    assert infer_anima_num_layers(_synthetic_block_state_dict(28)) == 28


def test_infer_anima_num_layers_expanded_40_net_prefix() -> None:
    wrapped = _synthetic_block_state_dict(40, prefix="net.")
    canonical = _strip_wrapping_prefixes(wrapped)
    assert infer_anima_num_layers(canonical) == 40


def test_infer_anima_num_layers_expanded_40_comfyui_prefix() -> None:
    wrapped = _synthetic_block_state_dict(40, prefix="model.diffusion_model.")
    canonical = _strip_wrapping_prefixes(wrapped)
    assert infer_anima_num_layers(canonical) == 40


def test_infer_anima_num_layers_rejects_gapped_blocks() -> None:
    state = _synthetic_block_state_dict(40)
    state.pop("blocks.17.self_attn.q_proj.weight")
    import pytest
    with pytest.raises(RuntimeError, match="not contiguous"):
        infer_anima_num_layers(state)


def test_infer_text_encoder_family_qwen3() -> None:
    from diffusers_anima.pipelines.anima.loading import infer_anima_text_encoder_family

    state = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(1)}
    assert infer_anima_text_encoder_family(state) == "qwen3"


def test_infer_text_encoder_family_qwen35_full() -> None:
    from diffusers_anima.pipelines.anima.loading import infer_anima_text_encoder_family

    state = {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(1)}
    assert infer_anima_text_encoder_family(state) == "qwen3.5"


def test_extract_qwen35_causal_state_dict_drops_non_text_modules() -> None:
    embed = torch.zeros(2, 2)
    raw = {
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(1),
        "model.visual.blocks.0.attn.qkv.weight": torch.ones(1),
        "mtp.layers.0.self_attn.q_proj.weight": torch.ones(1),
    }

    result = _extract_qwen35_causal_state_dict(raw)

    assert "model.embed_tokens.weight" in result
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in result
    assert "lm_head.weight" in result
    assert result["lm_head.weight"] is embed
    assert not any("visual" in key or key.startswith("mtp.") for key in result)
