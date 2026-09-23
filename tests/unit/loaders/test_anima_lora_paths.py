"""Anima LoRA module-path compatibility tests."""
from types import SimpleNamespace

import pytest
import torch

from diffusers_anima.loaders.lora_pipeline import (
    _remap_legacy_anima_lora_for_transformer,
)
from diffusers_anima.loaders.lora_pipeline import _map_anima_module_path
from diffusers_anima.models import ANIMA_29B_BASE_TO_EXPANDED


def test_maps_net_prefixed_expanded_block() -> None:
    assert (
        _map_anima_module_path("net.blocks.39.cross_attn.q_proj")
        == "core.transformer_blocks.39.attn2.to_q"
    )


def test_maps_comfyui_prefixed_expanded_block() -> None:
    assert (
        _map_anima_module_path("model.diffusion_model.blocks.39.mlp.layer2")
        == "core.transformer_blocks.39.ff.net.2"
    )


def test_maps_stacked_net_and_comfyui_prefixes() -> None:
    assert (
        _map_anima_module_path("model.diffusion_model.net.blocks.36.self_attn.output_proj")
        == "core.transformer_blocks.36.attn1.to_out.0"
    )


def _transformer(depth: int):
    return SimpleNamespace(config=SimpleNamespace(num_layers=depth))


def _lora_key(block_index: int) -> str:
    return f"transformer.core.transformer_blocks.{block_index}.attn2.to_q.lora_A.weight"


def test_remaps_complete_base_lora_to_29b_inherited_blocks() -> None:
    state_dict = {_lora_key(i): torch.tensor(float(i)) for i in range(28)}
    remapped, changed = _remap_legacy_anima_lora_for_transformer(
        state_dict, transformer=_transformer(40)
    )
    assert changed is True
    assert {
        int(key.split("transformer_blocks.", 1)[1].split(".", 1)[0])
        for key in remapped
    } == set(ANIMA_29B_BASE_TO_EXPANDED)
    assert remapped[_lora_key(35)].item() == 24.0


def test_partial_base_range_lora_requires_explicit_layout_on_29b() -> None:
    with pytest.raises(ValueError, match="Ambiguous partial LoRA"):
        _remap_legacy_anima_lora_for_transformer(
            {_lora_key(3): torch.tensor(1.0)}, transformer=_transformer(40)
        )


def test_explicit_partial_base_lora_is_remapped() -> None:
    remapped, changed = _remap_legacy_anima_lora_for_transformer(
        {_lora_key(3): torch.tensor(1.0)},
        transformer=_transformer(40),
        layout="base28",
    )
    assert changed is True
    assert set(remapped) == {_lora_key(4)}


def test_native_29b_lora_is_not_remapped() -> None:
    state_dict = {_lora_key(39): torch.tensor(1.0)}
    remapped, changed = _remap_legacy_anima_lora_for_transformer(
        state_dict, transformer=_transformer(40)
    )
    assert changed is False
    assert remapped is state_dict
