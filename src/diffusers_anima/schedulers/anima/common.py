from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import math
import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin

from ...pipelines.anima.constants import ANIMA_SAMPLING_MULTIPLIER
from ...pipelines.anima.image_processing import _ensure_finite


GeneratorInput = torch.Generator | list[torch.Generator] | tuple[torch.Generator, ...]


def randn_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    target_device = torch.device(device)

    if generator is None:
        return torch.randn(shape, device=target_device, dtype=dtype)

    if isinstance(generator, list):
        if shape[0] != len(generator):
            raise ValueError(
                f"`generator` list length must match tensor batch size ({shape[0]}), got {len(generator)}."
            )
        samples = []
        for item_generator in generator:
            sample = randn_tensor(
                (1, *shape[1:]),
                device=target_device,
                dtype=dtype,
                generator=item_generator,
            )
            samples.append(sample)
        return torch.cat(samples, dim=0)

    generator_device = (
        generator.device.type if hasattr(generator, "device") else target_device.type
    )
    if generator_device == target_device.type:
        return torch.randn(shape, device=target_device, dtype=dtype, generator=generator)

    noise = torch.randn(shape, device=generator.device, dtype=torch.float32, generator=generator)
    return noise.to(device=target_device, dtype=dtype)


def randn_like(
    sample: torch.Tensor,
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    return randn_tensor(
        tuple(sample.shape),
        device=sample.device,
        dtype=sample.dtype,
        generator=generator,
    )


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while x.ndim < target_ndim:
        x = x.unsqueeze(-1)
    return x


def sanitize_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    sigmas = sigmas.detach().to(dtype=torch.float32)
    sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    if sigmas.numel() == 0:
        return sigmas

    if sigmas[0] < sigmas[-1]:
        sigmas = torch.flip(sigmas, dims=[0])

    fixed = sigmas.clone()
    for i in range(1, fixed.numel()):
        if fixed[i] > fixed[i - 1]:
            fixed[i] = fixed[i - 1]

    if fixed[-1].item() != 0.0:
        fixed = torch.cat([fixed, fixed.new_zeros(1)], dim=0)

    return fixed


def to_d(latents: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    sigma = append_dims(sigma.to(device=latents.device, dtype=latents.dtype), latents.ndim)
    sigma = sigma.clamp_min(1e-8)
    return (latents - denoised) / sigma


def get_ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_from_f = sigma_from.float()
    sigma_to_f = sigma_to.float()

    if eta <= 0.0:
        return sigma_to_f, torch.zeros_like(sigma_to_f)

    sigma_up = torch.minimum(
        sigma_to_f,
        eta
        * torch.sqrt(
            torch.clamp(
                sigma_to_f**2 * (sigma_from_f**2 - sigma_to_f**2) / torch.clamp(sigma_from_f**2, min=1e-12),
                min=0.0,
            )
        ),
    )
    sigma_down = torch.sqrt(torch.clamp(sigma_to_f**2 - sigma_up**2, min=0.0))
    return sigma_down, sigma_up


def predict_noise_cfg(
    transformer: "ModelMixin",
    latents: torch.Tensor,
    *,
    model_timestep: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    model_input = latents.to(dtype=model_dtype)
    timestep = model_timestep.to(device=model_input.device, dtype=torch.float32)
    if timestep.ndim == 0:
        timestep = timestep.expand(model_input.shape[0])

    if cfg_batch_mode == "concat":
        model_input = torch.cat([model_input, model_input], dim=0)
        timestep = torch.cat([timestep, timestep], dim=0)
        encoder_hidden_states = torch.cat(
            [
                pos_cond.to(device=model_input.device, dtype=model_dtype),
                neg_cond.to(device=model_input.device, dtype=model_dtype),
            ],
            dim=0,
        )
        with torch.inference_mode():
            noise_pred = transformer(
                model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2, dim=0)

    elif cfg_batch_mode == "split":
        with torch.inference_mode():
            noise_pred_uncond = transformer(
                model_input,
                timestep,
                encoder_hidden_states=neg_cond.to(device=model_input.device, dtype=model_dtype),
                return_dict=False,
            )[0]
            noise_pred_text = transformer(
                model_input,
                timestep,
                encoder_hidden_states=pos_cond.to(device=model_input.device, dtype=model_dtype),
                return_dict=False,
            )[0]
    else:
        raise ValueError("cfg_batch_mode must be one of: split, concat.")

    noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise = noise.float()
    _ensure_finite(noise, name="noise prediction", runtime_dtype=torch.float32)
    return noise


def predict_denoised_const(
    transformer: "ModelMixin",
    latents: torch.Tensor,
    *,
    sigma: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    model_timestep = (
        (sigma * ANIMA_SAMPLING_MULTIPLIER).expand(latents.shape[0]).float()
    )
    noise_pred = predict_noise_cfg(
        transformer,
        latents,
        model_timestep=model_timestep,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        guidance_scale=guidance_scale,
        cfg_batch_mode=cfg_batch_mode,
        model_dtype=model_dtype,
    )
    denoised = latents.float() - append_dims(sigma.float(), latents.ndim) * noise_pred
    _ensure_finite(denoised, name="denoised prediction", runtime_dtype=torch.float32)
    return denoised


def run_step_callback(
    pipeline: "DiffusionPipeline",
    *,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    step_index: int,
    timestep: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    if callback_on_step_end is None:
        return latents

    callback_kwargs: dict[str, Any] = {}
    if "latents" in callback_on_step_end_tensor_inputs:
        callback_kwargs["latents"] = latents

    callback_outputs = callback_on_step_end(pipeline, step_index, timestep, callback_kwargs)
    if callback_outputs is None:
        return latents
    if not isinstance(callback_outputs, dict):
        raise TypeError("callback_on_step_end must return dict[str, Any] or None.")

    return callback_outputs.pop("latents", latents)


def apply_inpaint_source(
    latents: torch.Tensor,
    *,
    sigma_next: torch.Tensor,
    inpaint_mask: torch.Tensor | None,
    init_image_latents: torch.Tensor | None,
    init_noise: torch.Tensor | None,
) -> torch.Tensor:
    if inpaint_mask is None or init_image_latents is None or init_noise is None:
        return latents

    sigma_next_value = append_dims(
        sigma_next.to(device=init_image_latents.device, dtype=init_image_latents.dtype),
        init_image_latents.ndim,
    )
    source_latents = sigma_next_value * init_noise + (1.0 - sigma_next_value) * init_image_latents
    return (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents