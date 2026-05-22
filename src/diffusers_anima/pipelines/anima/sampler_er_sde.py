"""ER-SDE sampler for Anima constant-sigma sampling.

This module intentionally contains only the Extended Reverse-Time SDE sampler.
The generic CFG prediction, callback and RNG helpers stay in ``sampling.py`` and
are injected by ``run_const_sigma_samplers`` to keep this file independent from
AnimaFlowMatchEulerDiscreteScheduler and avoid circular imports.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin

import torch


PredictDenoisedFn = Callable[..., torch.Tensor]
RunStepCallbackFn = Callable[..., torch.Tensor]
RandnLikeFn = Callable[[torch.Tensor, torch.Generator | list[torch.Generator] | None], torch.Tensor]

_RF_SNR_MIN_SIGMA = 1e-6
_RF_SNR_MAX_SIGMA = 1.0 - 1e-3
_ER_SDE_NUM_INTEGRATION_POINTS = 200


def _offset_sigmas_for_er_sde_rf_snr(sigmas: torch.Tensor) -> torch.Tensor:
    """Avoid RF SNR singularities at sigma=1 and sigma=0.

    ForgeNEO offsets the first sigma before converting to SNR space. Anima's
    constant-sigma samplers use RF-style ``x = sigma * noise + (1 - sigma) * x0``
    values, so sigma=1 maps to alpha=0 and infinite lambda. Keeping this clamp
    local to ER-SDE avoids changing public sigma schedule builders.
    """
    return sigmas.to(dtype=torch.float32).clamp(
        min=0.0,
        max=_RF_SNR_MAX_SIGMA,
    )


def _sigma_to_rf_half_log_snr(sigmas: torch.Tensor) -> torch.Tensor:
    """RF half-log-SNR: log(alpha / sigma), alpha = 1 - sigma."""
    safe_sigma = sigmas.clamp(
        min=_RF_SNR_MIN_SIGMA,
        max=_RF_SNR_MAX_SIGMA,
    )
    alpha = (1.0 - safe_sigma).clamp_min(_RF_SNR_MIN_SIGMA)
    return torch.log(alpha) - torch.log(safe_sigma)


def _default_er_sde_noise_scaler(er_lambda: torch.Tensor) -> torch.Tensor:
    return er_lambda * ((er_lambda.pow(0.3)).exp() + 10.0)


def _safe_scalar_divisor(value: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError("Internal error: ER-SDE divisor must be a scalar tensor.")
    sign = torch.where(value < 0, value.new_tensor(-1.0), value.new_tensor(1.0))
    return sign * value.abs().clamp_min(eps)


def sample_er_sde(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor | None,
    guidance_scale: float,
    s_noise: float,
    generator: torch.Generator | list[torch.Generator] | None,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    check_finite: bool = False,
    er_sde_max_stage: int = 3,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
    predict_denoised: PredictDenoisedFn,
    run_step_callback: RunStepCallbackFn,
    randn_like_fn: RandnLikeFn,
) -> torch.Tensor:
    """Extended Reverse-Time SDE sampler adapted to Anima's RF sigma parameterization.

    This follows the ForgeNEO ER-SDE update structure, but replaces
    ``sigma_to_half_log_snr`` with the RF relation ``alpha = 1 - sigma`` and
    uses the pipeline's CFG prediction helper for the denoised estimate.
    """
    if er_sde_max_stage < 1 or er_sde_max_stage > 3:
        raise ValueError("`er_sde_max_stage` must be one of: 1, 2, 3.")
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError(
            "inpaint sampling requires both `init_image_latents` and `init_noise`."
        )

    sigmas = _offset_sigmas_for_er_sde_rf_snr(sigmas).to(device=latents.device)
    half_log_snrs = _sigma_to_rf_half_log_snr(sigmas)
    er_lambdas = half_log_snrs.neg().exp()
    point_indices = torch.arange(
        0,
        _ER_SDE_NUM_INTEGRATION_POINTS,
        dtype=torch.float32,
        device=latents.device,
    )

    old_denoised: torch.Tensor | None = None
    old_denoised_d: torch.Tensor | None = None

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = predict_denoised(
            transformer,
            latents,
            sigma=sigma,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
            check_finite=check_finite,
        )

        if i == len(sigmas) - 2:
            latents = denoised
            latents = run_step_callback(
                pipeline,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                step_index=i,
                timestep=sigma,
                latents=latents,
            )
            old_denoised = denoised
            continue

        stage_used = min(er_sde_max_stage, i + 1)
        er_lambda_s = er_lambdas[i]
        er_lambda_t = er_lambdas[i + 1]
        alpha_s = sigma / er_lambda_s
        alpha_t = sigma_next / er_lambda_t
        r_alpha = alpha_t / alpha_s
        r = _default_er_sde_noise_scaler(er_lambda_t) / _default_er_sde_noise_scaler(
            er_lambda_s
        )

        latents = (
            r_alpha.to(latents.dtype) * r.to(latents.dtype) * latents
            + alpha_t.to(latents.dtype) * (1.0 - r).to(latents.dtype) * denoised
        )

        if stage_used >= 2 and old_denoised is not None:
            dt = er_lambda_t - er_lambda_s
            lambda_step_size = -dt / float(_ER_SDE_NUM_INTEGRATION_POINTS)
            lambda_pos = er_lambda_t + point_indices * lambda_step_size
            scaled_pos = _default_er_sde_noise_scaler(lambda_pos).clamp_min(1e-20)

            integral_s = torch.sum(1.0 / scaled_pos) * lambda_step_size
            denom_d = _safe_scalar_divisor(er_lambda_s - er_lambdas[i - 1])
            denoised_d = (denoised - old_denoised) / denom_d.to(denoised.dtype)
            latents = latents + (
                alpha_t
                * (dt + integral_s * _default_er_sde_noise_scaler(er_lambda_t))
            ).to(latents.dtype) * denoised_d

            if stage_used >= 3 and old_denoised_d is not None:
                integral_u = (
                    torch.sum((lambda_pos - er_lambda_s) / scaled_pos)
                    * lambda_step_size
                )
                denom_u = _safe_scalar_divisor(
                    (er_lambda_s - er_lambdas[i - 2]) / 2.0
                )
                denoised_u = (denoised_d - old_denoised_d) / denom_u.to(
                    denoised_d.dtype
                )
                latents = latents + (
                    alpha_t
                    * (
                        (dt.square() / 2.0)
                        + integral_u * _default_er_sde_noise_scaler(er_lambda_t)
                    )
                ).to(latents.dtype) * denoised_u

            old_denoised_d = denoised_d

        if s_noise > 0:
            noise_variance = er_lambda_t.square() - er_lambda_s.square() * r.square()
            noise_coeff = noise_variance.clamp_min(0.0).sqrt().nan_to_num(nan=0.0)
            noise = randn_like_fn(latents, generator=generator)
            latents = latents + (
                alpha_t.to(latents.dtype)
                * noise
                * float(s_noise)
                * noise_coeff.to(latents.dtype)
            )

        if (
            inpaint_mask is not None
            and init_image_latents is not None
            and init_noise is not None
        ):
            sigma_next_value = sigma_next.to(init_image_latents.dtype)
            source_latents = (
                sigma_next_value * init_noise
                + (1.0 - sigma_next_value) * init_image_latents
            )
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents

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
