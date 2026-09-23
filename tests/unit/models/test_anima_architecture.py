"""Architecture-aware Anima quality profile tests."""

from types import SimpleNamespace

import torch

from diffusers_anima.models import (
    ANIMA_29B_BASE_TO_EXPANDED,
    ANIMA_29B_INSERTED_BLOCKS,
    anima_variant_from_num_layers,
    get_anima_transformer_num_layers,
)
from diffusers_anima.pipelines.anima.loading import (
    recommended_sampling_config_for_transformer,
)
from diffusers_anima.pipelines.anima.pipeline_anima import (
    _resolve_effective_cfg_batch_mode,
    _resolve_sample_dtype,
)


def _transformer(depth: int):
    return SimpleNamespace(config=SimpleNamespace(num_layers=depth))


def test_29b_mapping_partitions_all_40_blocks() -> None:
    assert len(ANIMA_29B_BASE_TO_EXPANDED) == 28
    assert len(ANIMA_29B_INSERTED_BLOCKS) == 12
    assert set(ANIMA_29B_BASE_TO_EXPANDED).isdisjoint(ANIMA_29B_INSERTED_BLOCKS)
    assert set(ANIMA_29B_BASE_TO_EXPANDED) | ANIMA_29B_INSERTED_BLOCKS == set(range(40))


def test_detects_architecture_and_recommended_sampler() -> None:
    assert get_anima_transformer_num_layers(_transformer(28)) == 28
    assert anima_variant_from_num_layers(40) == "2.9b"
    assert recommended_sampling_config_for_transformer(_transformer(28)) == (
        "euler_a_rf",
        "beta",
    )
    assert recommended_sampling_config_for_transformer(_transformer(40)) == (
        "euler",
        "uniform",
    )


def test_auto_sample_dtype_preserves_float32_latent_integration() -> None:
    assert _resolve_sample_dtype(
        "auto", model_dtype=torch.bfloat16, execution_device="cuda"
    ) == torch.float32
    assert _resolve_sample_dtype(
        "bfloat16", model_dtype=torch.bfloat16, execution_device="cuda"
    ) == torch.bfloat16


def test_29b_auto_cfg_uses_reference_split_path() -> None:
    assert _resolve_effective_cfg_batch_mode(
        "auto", execution_device="cuda", transformer_num_layers=40
    ) == "split"
    assert _resolve_effective_cfg_batch_mode(
        "auto", execution_device="cuda", transformer_num_layers=28
    ) == "concat"
