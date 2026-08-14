"""Anima LoRA module-path compatibility tests."""
from diffusers_anima.loaders.lora_pipeline import _map_anima_module_path


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
