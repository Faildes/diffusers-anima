from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin

from .common import (
    randn_like,
    sanitize_sigmas,
    predict_denoised_const,
    run_step_callback,
    apply_inpaint_source,
)
from ...pipelines.anima.image_processing import _ensure_finite


def sample_euler_ancestral(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    eta: float,
    s_noise: float,
    generator: torch.Generator | list[torch.Generator] | None,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError(
            "inpaint sampling requires both `init_image_latents` and `init_noise`."
        )

    sigmas = sanitize_sigmas(sigmas)
    latents = latents.float()

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(latents, sigma)

        if sigma_next.item() == 0.0:
            latents = denoised
        else:
            sigma_f = sigma.float().clamp_min(1e-8)
            sigma_next_f = sigma_next.float()

            downstep_ratio = 1.0 + (sigma_next_f / sigma_f - 1.0) * float(eta)
            sigma_down = sigma_next_f * downstep_ratio

            alpha_ip1 = (1.0 - sigma_next_f).clamp_min(1e-8)
            alpha_down = (1.0 - sigma_down).clamp_min(1e-8)

            renoise_sq = sigma_next_f**2 - (
                sigma_down**2 * alpha_ip1**2 / alpha_down**2
            )
            renoise_coeff = torch.sqrt(torch.clamp(renoise_sq, min=0.0))

            sigma_down_ratio = sigma_down / sigma_f
            while sigma_down_ratio.ndim < latents.ndim:
                sigma_down_ratio = sigma_down_ratio.unsqueeze(-1)

            latents = (
                sigma_down_ratio.to(latents.dtype) * latents
                + (1.0 - sigma_down_ratio.to(latents.dtype)) * denoised
            )

            if eta > 0.0 and float(renoise_coeff.item()) > 0.0:
                noise = randn_like(latents, generator=generator).float()

                alpha_ratio = (alpha_ip1 / alpha_down).to(torch.float32)
                while alpha_ratio.ndim < latents.ndim:
                    alpha_ratio = alpha_ratio.unsqueeze(-1)

                renoise_coeff_v = renoise_coeff.to(torch.float32)
                while renoise_coeff_v.ndim < latents.ndim:
                    renoise_coeff_v = renoise_coeff_v.unsqueeze(-1)

                latents = (
                    alpha_ratio.to(latents.dtype) * latents
                    + noise * float(s_noise) * renoise_coeff_v.to(latents.dtype)
                )

            latents = apply_inpaint_source(
                latents,
                sigma_next=sigma_next,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )

        _ensure_finite(
            latents,
            name="latents after Euler ancestral RF step",
            runtime_dtype=torch.float32,
        )

        latents = run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )

    return latents