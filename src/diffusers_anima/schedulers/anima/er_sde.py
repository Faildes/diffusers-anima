from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin

from .common import (
    randn_like,
    sanitize_sigmas,
    to_d,
    get_ancestral_step,
    predict_denoised_const,
    run_step_callback,
    apply_inpaint_source,
)
from ...pipelines.anima.image_processing import _ensure_finite


def sample_er_sde(
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
    solver_type: str = "er_sde",
    max_stage: int = 3,
) -> torch.Tensor:
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    sigmas = sanitize_sigmas(sigmas)
    latents = latents.float()
    max_stage = max(1, min(int(max_stage), 3))
    solver_type = str(solver_type).lower()

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = predict_denoised_const(
            transformer,
            latents,
            sigma=sigma,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
        )

        if sigma_next.item() == 0.0:
            latents = denoised
        else:
            d0 = to_d(latents, sigma, denoised)

            if solver_type == "ode":
                dt = (sigma_next - sigma).to(torch.float32)
                while dt.ndim < latents.ndim:
                    dt = dt.unsqueeze(-1)
                x_pred = latents + d0 * dt
            else:
                sigma_down, sigma_up = get_ancestral_step(
                    sigma,
                    sigma_next,
                    eta=1.0 if solver_type == "er_sde" else eta,
                )
                dt = (sigma_down - sigma).to(torch.float32)
                while dt.ndim < latents.ndim:
                    dt = dt.unsqueeze(-1)
                x_pred = latents + d0 * dt

                if float(sigma_up.item()) > 0.0:
                    noise = randn_like(latents, generator=generator).float()
                    sigma_up_v = sigma_up.to(torch.float32)
                    if solver_type == "reverse_time_sde":
                        sigma_up_v = sigma_up_v ** (eta + 1.0)
                    while sigma_up_v.ndim < latents.ndim:
                        sigma_up_v = sigma_up_v.unsqueeze(-1)
                    x_pred = x_pred + noise * s_noise * sigma_up_v

            latents_new = x_pred
            for _ in range(max_stage - 1):
                den2 = predict_denoised_const(
                    transformer,
                    latents_new,
                    sigma=sigma_next,
                    pos_cond=pos_cond,
                    neg_cond=neg_cond,
                    guidance_scale=guidance_scale,
                    cfg_batch_mode=cfg_batch_mode,
                    model_dtype=model_dtype,
                )
                d1 = to_d(latents_new, sigma_next, den2)
                dt = (sigma_next - sigma).to(torch.float32)
                while dt.ndim < latents.ndim:
                    dt = dt.unsqueeze(-1)
                latents_new = latents + 0.5 * (d0 + d1) * dt

            latents = latents_new
            latents = apply_inpaint_source(
                latents,
                sigma_next=sigma_next,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )

        _ensure_finite(latents, name="latents after ER-SDE step", runtime_dtype=torch.float32)

        latents = run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )

    return latents