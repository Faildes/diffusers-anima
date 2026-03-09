from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin

from .common import (
    randn_like,
    sanitize_sigmas,
    offset_first_sigma_for_snr_const,
    sigma_to_half_log_snr_const,
    default_er_sde_noise_scaler,
    predict_denoised_const,
    run_step_callback,
    apply_inpaint_source,
    _safe_signed_denom,
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
    solver_type: str = "midpoint",
    max_stage: int = 3,
) -> torch.Tensor:
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError(
            "inpaint sampling requires both `init_image_latents` and `init_noise`."
        )

    sigmas = sanitize_sigmas(sigmas)
    sigmas = offset_first_sigma_for_snr_const(sigmas)

    latents = latents.float()
    max_stage = max(1, min(int(max_stage), 3))
    solver_type = str(solver_type).strip().lower()

    deterministic = solver_type == "ode" or float(s_noise) <= 0.0

    num_integration_points = 200.0
    point_indices = torch.arange(
        0,
        num_integration_points,
        dtype=torch.float32,
        device=latents.device,
    )

    half_log_snrs = sigma_to_half_log_snr_const(sigmas)
    er_lambdas = torch.exp(-half_log_snrs)  # sigma / alpha for CONST case

    old_denoised: torch.Tensor | None = None
    old_denoised_d: torch.Tensor | None = None

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(latents, sigma)

        stage_used = min(max_stage, i + 1)

        if sigma_next.item() == 0.0:
            latents = denoised
        else:
            er_lambda_s = er_lambdas[i]
            er_lambda_t = er_lambdas[i + 1]

            alpha_s = sigma.float() / er_lambda_s.clamp_min(1e-12)
            alpha_t = sigma_next.float() / er_lambda_t.clamp_min(1e-12)

            scaled_s = default_er_sde_noise_scaler(er_lambda_s).clamp_min(1e-12)
            scaled_t = default_er_sde_noise_scaler(er_lambda_t).clamp_min(1e-12)

            r_alpha = alpha_t / alpha_s.clamp_min(1e-12)
            r = scaled_t / scaled_s

            latents = (
                r_alpha.to(latents.dtype) * r.to(latents.dtype) * latents
                + alpha_t.to(latents.dtype) * (1.0 - r).to(latents.dtype) * denoised
            )

            if stage_used >= 2 and old_denoised is not None:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indices * lambda_step_size
                scaled_pos = default_er_sde_noise_scaler(lambda_pos).clamp_min(1e-12)

                s = torch.sum(1.0 / scaled_pos) * lambda_step_size

                prev_gap = _safe_signed_denom(er_lambda_s - er_lambdas[i - 1])
                denoised_d = (denoised - old_denoised) / prev_gap

                latents = latents + (
                    alpha_t.to(latents.dtype)
                    * (dt + s * scaled_t).to(latents.dtype)
                    * denoised_d.to(latents.dtype)
                )

                if stage_used >= 3 and old_denoised_d is not None and i >= 2:
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    prev2_gap = _safe_signed_denom((er_lambda_s - er_lambdas[i - 2]) / 2.0)
                    denoised_u = (denoised_d - old_denoised_d) / prev2_gap

                    latents = latents + (
                        alpha_t.to(latents.dtype)
                        * (((dt**2) / 2.0) + s_u * scaled_t).to(latents.dtype)
                        * denoised_u.to(latents.dtype)
                    )

                old_denoised_d = denoised_d

            if not deterministic:
                noise = randn_like(latents, generator=generator).float()
                noise_scale_sq = er_lambda_t**2 - er_lambda_s**2 * r**2
                noise_scale = torch.sqrt(torch.clamp(noise_scale_sq, min=0.0))
                latents = latents + (
                    alpha_t.to(latents.dtype)
                    * noise
                    * float(s_noise)
                    * noise_scale.to(latents.dtype)
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
            name="latents after ER-SDE step",
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

        old_denoised = denoised

    return latents