"""Seed batching reuses precomputed prompt embeds without changing CPU RNG."""

from types import SimpleNamespace

import pytest
import torch

from diffusers_anima.pipelines.anima.pipeline_anima import (
    AnimaPipeline,
    _expand_conditioning_batch,
)
from diffusers_anima.pipelines.anima.sampling import (
    _precompute_cpu_ancestral_noise,
    randn_tensor,
)


def test_precomputed_conditioning_expands_by_prompt_then_seed():
    positive = torch.tensor([[[1.0]], [[2.0]]])
    negative = torch.tensor([[[-1.0]], [[-2.0]]])
    expanded_pos, expanded_neg = _expand_conditioning_batch(
        positive, negative, num_images_per_prompt=3
    )
    assert expanded_pos[:, 0, 0].tolist() == [1, 1, 1, 2, 2, 2]
    assert expanded_neg[:, 0, 0].tolist() == [-1, -1, -1, -2, -2, -2]
    unchanged_pos, unchanged_neg = _expand_conditioning_batch(
        positive, negative, num_images_per_prompt=1
    )
    assert unchanged_pos is positive
    assert unchanged_neg is negative


def test_cpu_generator_list_matches_individual_seed_sequences():
    seeds = [7, 11, 19]
    generators = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    init_batch = randn_tensor((3, 16, 1, 4, 4), device="cpu", dtype=torch.float32, generator=generators)
    step_batch = randn_tensor((3, 16, 1, 4, 4), device="cpu", dtype=torch.float32, generator=generators)
    for i, seed in enumerate(seeds):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = randn_tensor((1, 16, 1, 4, 4), device="cpu", dtype=torch.float32, generator=generator)
        step = randn_tensor((1, 16, 1, 4, 4), device="cpu", dtype=torch.float32, generator=generator)
        assert torch.equal(init_batch[i : i + 1], initial)
        assert torch.equal(step_batch[i : i + 1], step)


def test_prefetched_cpu_noise_keeps_draw_order_and_generator_state():
    seeds = [7, 11]
    generators = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    reference = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    latents = torch.empty((2, 16, 1, 4, 4), dtype=torch.float32)
    cached = _precompute_cpu_ancestral_noise(latents, count=3, generator=generators)
    for step in range(3):
        expected = randn_tensor(
            tuple(latents.shape), device="cpu", dtype=torch.float32,
            generator=reference,
        )
        assert torch.equal(cached[step], expected)
    for got, original in zip(generators, reference):
        assert torch.equal(got.get_state(), original.get_state())


def test_check_inputs_uses_expanded_embed_batch_for_generators():
    pipe = SimpleNamespace(spatial_step=16, _callback_tensor_inputs=["latents"])
    embeds = torch.zeros((1, 512, 1024))
    common = dict(
        prompt=None, negative_prompt=None,
        prompt_embeds=embeds, negative_prompt_embeds=embeds,
        image=None, mask_image=None, strength=1.0,
        width=512, height=512, num_inference_steps=2,
        num_images_per_prompt=3, sampler="euler_a_rf", sigma_schedule="normal",
        cfg_batch_mode="concat", output_type="pil",
    )
    generators = [torch.Generator(device="cpu").manual_seed(i) for i in range(3)]
    AnimaPipeline.check_inputs(pipe, generator=generators, **common)
    with pytest.raises(ValueError, match="batch size"):
        AnimaPipeline.check_inputs(pipe, generator=generators[:2], **common)
    with pytest.raises(ValueError, match="matching shapes"):
        AnimaPipeline.check_inputs(
            pipe, generator=generators,
            **{**common, "negative_prompt_embeds": embeds.repeat(2, 1, 1)},
        )
