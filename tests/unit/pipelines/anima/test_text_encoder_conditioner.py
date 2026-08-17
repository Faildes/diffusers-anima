import torch

from diffusers_anima.pipelines.anima.text_encoder_bridge import AnimaTextEncoderBridge
from diffusers_anima.pipelines.anima.text_encoder_conditioner import AnimaTextEncoderConditioner


def _conditioner(strength=0.25):
    bridge = AnimaTextEncoderBridge(
        rotation=torch.eye(4),
        source_mean=torch.zeros(4),
        target_mean=torch.zeros(4),
        variance_scale=torch.tensor([2.0, 1.0, 1.0, 1.0]),
        center_strength=1.0,
        variance_strength=1.0,
        metadata={"format": "anima_text_encoder_profile_v2"},
    )
    return AnimaTextEncoderConditioner(
        alignment=bridge,
        semantic_expansion_strength=strength,
        semantic_expansion_max_tokens=2,
        semantic_expansion_chunk_size=2,
        semantic_expansion_min_source_tokens=2,
        semantic_expansion_residual_clip=1.0,
    )


def test_zero_expansion_preserves_primary_contract():
    cond = _conditioner(0.0)
    x = torch.randn(1, 4, 4)
    mask = torch.ones(1, 4, dtype=torch.long)
    out, out_mask = cond.build_memory(x, mask)
    torch.testing.assert_close(out, cond.align(x))
    torch.testing.assert_close(out_mask, mask)


def test_semantic_expansion_appends_bounded_slots_without_replacing_tokens():
    cond = _conditioner(0.25)
    x = torch.randn(1, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    primary = cond.align(x)
    out, out_mask = cond.build_memory(x, mask)
    assert out.shape == (1, 7, 4)
    torch.testing.assert_close(out[:, :5], primary)
    assert out_mask.tolist() == [[1, 1, 1, 1, 0, 1, 1]]


def test_v4_group_aware_expansion_keeps_groups_as_separate_slots():
    cond = _conditioner(0.25)
    cond.semantic_expansion_chunk_size = 64
    cond.semantic_expansion_max_tokens = 4
    cond.semantic_expansion_group_aware = True
    cond.semantic_expansion_min_coherence = 0.0
    x = torch.randn(1, 4, 4)
    mask = torch.ones(1, 4, dtype=torch.long)
    groups = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    out, out_mask = cond.build_memory(x, mask, group_ids=groups)
    # One semantic slot for each contiguous PromptPlan group.
    assert out.shape == (1, 6, 4)
    assert out_mask.tolist() == [[1, 1, 1, 1, 1, 1]]


def test_v5_short_source_semantic_slots_do_not_cross_native_window():
    cond = _conditioner(0.25)
    cond.semantic_expansion_max_tokens = 16
    cond.semantic_expansion_chunk_size = 64
    cond.semantic_expansion_native_window = 512
    cond.semantic_expansion_preserve_native_window = True
    x = torch.randn(1, 511, 4)
    mask = torch.ones(1, 511, dtype=torch.long)
    out, out_mask = cond.build_memory(x, mask)
    assert out.shape[1] == 512
    assert out_mask.shape[1] == 512
