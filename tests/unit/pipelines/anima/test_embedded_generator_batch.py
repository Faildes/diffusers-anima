"""Regress precomputed SD_Embed conditioning with one CPU seed per image."""

from types import SimpleNamespace

import pytest
import torch

from diffusers_anima.pipelines.anima.pipeline_anima import (
    AnimaPipeline,
    _expand_conditioning_batch,
)


def test_embedded_generator_batch_validates_output_size():
    pipe = SimpleNamespace(spatial_step=16, _callback_tensor_inputs=["latents"])
    embeds = torch.zeros((1, 512, 1024))
    kwargs = dict(
        prompt=None, negative_prompt=None,
        prompt_embeds=embeds, negative_prompt_embeds=embeds,
        image=None, mask_image=None, strength=1.0,
        width=512, height=512, num_inference_steps=2,
        num_images_per_prompt=2, sampler="euler_a_rf", sigma_schedule="normal",
        cfg_batch_mode="concat", output_type="pil",
    )
    generators = [torch.Generator("cpu").manual_seed(seed) for seed in (7, 11)]
    AnimaPipeline.check_inputs(pipe, generator=generators, **kwargs)
    with pytest.raises(ValueError, match="batch size"):
        AnimaPipeline.check_inputs(pipe, generator=generators[:1], **kwargs)


def test_embedded_generator_batch_repeats_conditioning_in_prompt_order():
    positive = torch.tensor([[[1.0]], [[2.0]]])
    negative = torch.tensor([[[-1.0]], [[-2.0]]])
    pos, neg = _expand_conditioning_batch(
        positive, negative, num_images_per_prompt=2,
    )
    assert pos[:, 0, 0].tolist() == [1, 1, 2, 2]
    assert neg[:, 0, 0].tolist() == [-1, -1, -2, -2]
