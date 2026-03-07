from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import math
import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin

from .common import (
    randn_like,
    sanitize_sigmas,
    to_d,
    predict_denoised_const,
    run_step_callback,
    apply_inpaint_source,
)
from ...pipelines.anima.image_processing import _ensure_finite


def sample_euler(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> torch.Tensor:
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    sigmas = sanitize_sigmas(sigmas)
    latents = latents.float()

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = 0.0
        sigma_value = float(sigma.item())
        if s_churn > 0.0 and s_tmin <= sigma_value <= s_tmax:
            gamma = min(s_churn / max(len(sigmas) - 1, 1), math.sqrt(2.0) - 1.0)

        sigma_hat = sigma * (1.0 + gamma)

        if gamma > 0.0:
            eps = randn_like(latents, generator=generator).float()
            latents = latents + eps * s_noise * torch.sqrt(torch.clamp(sigma_hat**2 - sigma**2, min=0.0))

        denoised = predict_denoised_const(
            transformer,
            latents,
            sigma=sigma_hat,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
        )

        if sigma_next.item() == 0.0:
            latents = denoised
        else:
            d = to_d(latents, sigma_hat, denoised)
            dt = (sigma_next - sigma_hat).to(torch.float32)
            while dt.ndim < latents.ndim:
                dt = dt.unsqueeze(-1)
            latents = latents + d * dt
            latents = apply_inpaint_source(
                latents,
                sigma_next=sigma_next,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )

        _ensure_finite(latents, name="latents after Euler step", runtime_dtype=torch.float32)

        latents = run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )

    return latents