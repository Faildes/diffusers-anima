from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin, SchedulerMixin

from .common import predict_noise_cfg, run_step_callback, apply_inpaint_source
from ...pipelines.anima.constants import ANIMA_SAMPLING_MULTIPLIER
from ...pipelines.anima.image_processing import _ensure_finite


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
    if sigma_schedule != "uniform":
        raise ValueError("flowmatch_euler sampler only supports sigma_schedule='uniform'.")
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    _iterable = pipeline.progress_bar(timesteps)
    for i, timestep in enumerate(_iterable):
        scheduler_timestep = timestep.expand(latents.shape[0]).float()
        model_timestep = (
            scheduler_timestep / float(scheduler.config.num_train_timesteps)
        ) * ANIMA_SAMPLING_MULTIPLIER

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

        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0].float()
        _ensure_finite(latents, name="latents after flowmatch step", runtime_dtype=torch.float32)

        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            source_latents = init_image_latents
            if i < len(timesteps) - 1:
                next_timestep = (
                    timesteps[i + 1]
                    .expand(init_image_latents.shape[0])
                    .to(device=init_image_latents.device, dtype=torch.float32)
                )
                source_latents = scheduler.scale_noise(init_image_latents, next_timestep, init_noise)
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents

        latents = run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=timestep,
            latents=latents,
        )

    return latents