from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin, SchedulerMixin

import torch

from ...schedulers.anima import (
    randn_tensor,
    randn_like,
    sample_flowmatch_euler as _sample_flowmatch_euler,
    sample_euler as _sample_euler,
    sample_euler_ancestral as _sample_euler_ancestral,
    sample_er_sde as _sample_er_sde,
)


GeneratorInput = torch.Generator | list[torch.Generator] | tuple[torch.Generator, ...]


_LEGACY_SAMPLER_ALIASES = {
    "euler_a_rf": "euler_a",
    "euler_ancestral_rf": "euler_ancestral",
}


def _normalize_sampler_name(sampler: str) -> str:
    sampler = str(sampler).strip().lower()
    return _LEGACY_SAMPLER_ALIASES.get(sampler, sampler)


def sample_flowmatch_euler(
    transformer: "ModelMixin",
    scheduler: "SchedulerMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    timesteps: torch.Tensor,
    sigma_schedule: str,
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
) -> torch.Tensor:
    return _sample_flowmatch_euler(
        transformer,
        scheduler,
        pipeline,
        latents,
        timesteps=timesteps,
        sigma_schedule=sigma_schedule,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        guidance_scale=guidance_scale,
        cfg_batch_mode=cfg_batch_mode,
        model_dtype=model_dtype,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        inpaint_mask=inpaint_mask,
        init_image_latents=init_image_latents,
        init_noise=init_noise,
    )


def run_const_sigma_samplers(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    sampler: str,
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
    input_is_noisy_latents: bool = False,
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
    solver_type: str = "midpoint",
    max_stage: int = 2,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float | None = None,
) -> torch.Tensor:
    """
    Dispatch to the selected non-flowmatch sampler.

    Notes:
      - legacy aliases euler_a_rf / euler_ancestral_rf are normalized.
      - s_tmax=None is converted to +inf for samplers that expect a float range.
      - latents are promoted to float32 for numerical stability.
    """
    if len(sigmas) < 2:
        raise ValueError("At least 1 denoising step is required.")

    if not input_is_noisy_latents:
        latents = latents.float() * sigmas[0].to(torch.float32)
    else:
        latents = latents.float()

    sampler = _normalize_sampler_name(sampler)
    s_tmax_eff = float("inf") if s_tmax is None else float(s_tmax)

    if sampler == "euler":
        return _sample_euler(
            transformer,
            pipeline,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
            s_churn=float(s_churn),
            s_tmin=float(s_tmin),
            s_tmax=s_tmax_eff,
            s_noise=float(s_noise),
            generator=generator,
        )

    if sampler in {"euler_a", "euler_ancestral"}:
        return _sample_euler_ancestral(
            transformer,
            pipeline,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            eta=float(eta),
            s_noise=float(s_noise),
            generator=generator,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
        )

    if sampler == "er_sde":
        return _sample_er_sde(
            transformer,
            pipeline,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            eta=float(eta),
            s_noise=float(s_noise),
            generator=generator,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
            solver_type=str(solver_type).strip().lower(),
            max_stage=int(max_stage),
        )

    raise ValueError(
        f"Unsupported sampler '{sampler}'. Choose one of: "
        "flowmatch_euler, euler, euler_a, euler_ancestral, er_sde."
    )


__all__ = [
    "GeneratorInput",
    "randn_tensor",
    "randn_like",
    "sample_flowmatch_euler",
    "run_const_sigma_samplers",
]