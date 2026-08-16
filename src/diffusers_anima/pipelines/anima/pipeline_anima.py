"""Anima pipeline implementation with Diffusers-style loading conventions."""

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
import warnings

from transformers import PreTrainedModel

from diffusers import (
    AutoencoderKLQwenImage,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ...loaders.lora_pipeline import AnimaLoraLoaderMixin
from ...models.transformers.modeling_anima_transformer import AnimaTransformerModel
from ...schedulers import AnimaFlowMatchEulerDiscreteScheduler, AnimaSamplingConfig
from .constants import (
    DTYPE_MAP,
    FORGE_BETA_ALPHA,
    FORGE_BETA_BETA,
)
from .generator_utils import (
    _normalize_generator,
    _resolve_noise_runtime,
)
from .image_processing import (
    align_tensor_batch_size,
    decode_latents,
    encode_image_to_latents,
    latent_hw,
    prepare_init_image_tensor,
    prepare_inpaint_mask_tensor,
)
from .loading import (
    build_anima_pipeline,
    clear_anima_component_cache,
    coerce_anima_scheduler,
    load_prompt_tokenizer,
    loader_options_from_kwargs,
    normalize_loaded_component_buffers,
    resolve_patch_size,
    resolve_prompt_tokenizer_sources_for_local_dir,
    resolve_vae_scale_factor,
    runtime_options_from_kwargs,
    save_prompt_tokenizers_to_local_dir,
    scheduler_from_kwargs,
)
from .loading import (
    _disable_vae_method,
    _enable_vae_method,
)
from .options import AnimaComponents, AnimaLoaderOptions
from .pipeline_output import AnimaPipelineOutput
from .prompt_utils import _resolve_prompt_batches
from .sampling import GeneratorInput, randn_tensor, run_const_sigma_samplers, sample_flowmatch_euler
from .sigma_schedules import build_sampling_sigmas
from .strength_utils import (
    _trim_flowmatch_timesteps_by_strength,
    _trim_sigmas_by_strength,
)
from .text_encoding import (
    AnimaPromptTokenizer,
    build_condition,
    prepare_condition_inputs,
    prepare_condition_inputs_from_plans,
)
from .text_encoder_bridge import AnimaTextEncoderBridge
from .validation import (
    ImageBatchInput,
    PromptInput,
    _ANIMA_COMPONENT_OVERRIDE_KEYS,
    _DIFFUSERS_COMPAT_IGNORED_FROM_SINGLE_FILE_KEYS,
    _looks_like_single_file_source,
    _partition_single_file_from_pretrained_kwargs,
    _pop_ignored_kwargs,
    _raise_if_removed_from_pretrained_runtime_feature_kwargs,
    _validate_callback_tensor_input_names,
    _validate_image_like_input,
    _validate_sampling_modes,
    _warn_ignored_sampling_arguments,
)

@contextmanager
def _module_execution_context(
    module: torch.nn.Module,
    *,
    execution_device: str,
    execution_dtype: torch.dtype,
    enable_offload: bool,
) -> Iterator[None]:
    if enable_offload and execution_device != "cpu":
        module.to(device=execution_device, dtype=execution_dtype)
        try:
            yield
        finally:
            module.to(device="cpu")
            if execution_device == "cuda":
                torch.cuda.empty_cache()
        return

    yield


def _resolve_sample_dtype(
    sample_dtype: str | torch.dtype,
    *,
    model_dtype: torch.dtype,
    execution_device: str,
) -> torch.dtype:
    """Resolve the denoising latent dtype.

    Diffusers-style pipelines usually keep latents in the inference/model dtype on
    CUDA. The previous Anima path always used float32 latents, forcing a cast into
    ``model_dtype`` at every transformer step and then promoting the result back
    to float32. ``auto`` keeps the older stable float32 path on CPU, but uses the
    model dtype on CUDA/MPS when it is a common inference dtype.
    """
    if isinstance(sample_dtype, torch.dtype):
        return sample_dtype
    if not isinstance(sample_dtype, str):
        raise ValueError("`sample_dtype` must be 'auto', a dtype name, or torch.dtype.")
    mapped = DTYPE_MAP.get(sample_dtype)
    if sample_dtype != "auto" and mapped is None:
        raise ValueError(f"Unsupported sample_dtype: {sample_dtype}")
    if mapped is not None:
        return mapped
    if execution_device in {"cuda", "mps"} and model_dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        return model_dtype
    return torch.float32


def _resolve_effective_cfg_batch_mode(cfg_batch_mode: str, *, execution_device: str) -> str:
    """Resolve ``cfg_batch_mode='auto'`` to a concrete execution strategy."""
    if cfg_batch_mode == "auto":
        # CUDA benefits most from the Diffusers-style batched CFG forward. CPU/MPS
        # often prefer lower peak memory and avoid the doubled batch.
        return "concat" if execution_device == "cuda" else "split"
    return cfg_batch_mode


def _active_text_encoder_family(text_encoder: torch.nn.Module) -> str:
    tagged = getattr(text_encoder, "_anima_text_encoder_family", None)
    if tagged:
        return str(tagged)
    model_type = str(getattr(getattr(text_encoder, "config", None), "model_type", ""))
    if model_type == "qwen3_5_text":
        return "qwen3.5"
    if model_type == "qwen3":
        return "qwen3"
    return "unknown"


# ---------------------------------------------------------------------------
# Internal image generation routine
# ---------------------------------------------------------------------------


def _prepare_prompt_embedding_inputs(
    pipe: "AnimaPipeline",
    *,
    prompt: list[str],
    negative_prompt: list[str] | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
]:
    with _module_execution_context(
        pipe.text_encoder,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.text_encoder_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        if pipe.prompt_tokenizer is None:
            raise RuntimeError(
                "AnimaPipeline requires a prompt_tokenizer. "
                "Load the pipeline via from_pretrained, from_single_file, or "
                "from_multiple_files to ensure the tokenizer is initialised automatically."
            )
        family = _active_text_encoder_family(pipe.text_encoder)
        if family == "qwen3.5" and pipe.text_encoder_bridge is None and not getattr(pipe, "_anima_warned_missing_bridge", False):
            warnings.warn(
                "Qwen3.5 is shape-compatible with Anima but its hidden representation is not the "
                "Qwen3-0.6B space used to train the LLM adapter. Calibrate/load an "
                "AnimaTextEncoderBridge for production use.",
                stacklevel=2,
            )
            pipe._anima_warned_missing_bridge = True
        pos_hidden, pos_qwen_mask, pos_t5_ids, pos_t5_mask, pos_t5_weights = prepare_condition_inputs(
            pipe.prompt_tokenizer,
            pipe.text_encoder,
            prompt,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
            bridge=pipe.text_encoder_bridge,
        )
        if negative_prompt is None:
            return (
                pos_hidden, pos_qwen_mask, pos_t5_ids, pos_t5_mask, pos_t5_weights,
                None, None, None, None, None,
            )
        neg_hidden, neg_qwen_mask, neg_t5_ids, neg_t5_mask, neg_t5_weights = prepare_condition_inputs(
            pipe.prompt_tokenizer,
            pipe.text_encoder,
            negative_prompt,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
            bridge=pipe.text_encoder_bridge,
        )

    return (
        pos_hidden, pos_qwen_mask, pos_t5_ids, pos_t5_mask, pos_t5_weights,
        neg_hidden, neg_qwen_mask, neg_t5_ids, neg_t5_mask, neg_t5_weights,
    )


def _build_cfg_conditions_from_embeddings(
    pipe: "AnimaPipeline",
    *,
    pos_hidden: torch.Tensor,
    pos_qwen_mask: torch.Tensor | None,
    pos_t5_ids: torch.Tensor,
    pos_t5_mask: torch.Tensor | None,
    pos_t5_weights: torch.Tensor,
    neg_hidden: torch.Tensor | None,
    neg_qwen_mask: torch.Tensor | None,
    neg_t5_ids: torch.Tensor | None,
    neg_t5_mask: torch.Tensor | None,
    neg_t5_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pos_cond = build_condition(
        pipe.transformer,
        qwen_hidden=pos_hidden,
        qwen_mask=pos_qwen_mask,
        t5_ids=pos_t5_ids,
        t5_mask=pos_t5_mask,
        t5_weights=pos_t5_weights,
    )
    if neg_hidden is None or neg_t5_ids is None or neg_t5_weights is None:
        return pos_cond, None
    neg_cond = build_condition(
        pipe.transformer,
        qwen_hidden=neg_hidden,
        qwen_mask=neg_qwen_mask,
        t5_ids=neg_t5_ids,
        t5_mask=neg_t5_mask,
        t5_weights=neg_t5_weights,
    )
    return pos_cond, neg_cond


def _encode_prompt_plan_batch(
    pipe: "AnimaPipeline",
    plans: list[Any],
) -> torch.Tensor:
    if pipe.prompt_tokenizer is None:
        raise RuntimeError("AnimaPipeline requires a prompt_tokenizer for prompt plans.")
    with _module_execution_context(
        pipe.text_encoder,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.text_encoder_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        hidden, qwen_mask, t5_ids, t5_mask, t5_weights = prepare_condition_inputs_from_plans(
            pipe.prompt_tokenizer,
            pipe.text_encoder,
            plans,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
            bridge=pipe.text_encoder_bridge,
        )
    with _module_execution_context(
        pipe.transformer,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        return build_condition(
            pipe.transformer,
            qwen_hidden=hidden,
            qwen_mask=qwen_mask,
            t5_ids=t5_ids,
            t5_mask=t5_mask,
            t5_weights=t5_weights,
        )


def _prepare_init_image_latents_and_inpaint_mask(
    pipe: "AnimaPipeline",
    *,
    image: ImageBatchInput | None,
    mask_image: ImageBatchInput | None,
    width: int,
    height: int,
    latent_h: int,
    latent_w: int,
    batch_size: int,
    init_generator: torch.Generator | list[torch.Generator] | None,
    sample_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if image is None:
        return None, None

    init_image_tensor = prepare_init_image_tensor(
        image,
        width=width,
        height=height,
    )
    init_image_tensor = align_tensor_batch_size(
        init_image_tensor,
        target_batch_size=batch_size,
        input_name="image",
    )
    with _module_execution_context(
        pipe.vae,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        init_image_latents = encode_image_to_latents(
            pipe.vae,
            image_tensor=init_image_tensor,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
            generator=init_generator,
            sample_dtype=sample_dtype,
        )
    init_image_latents = init_image_latents.to(
        device=pipe.execution_device, dtype=sample_dtype
    )

    if tuple(init_image_latents.shape[-2:]) != (latent_h, latent_w):
        raise RuntimeError(
            "Encoded image latent shape does not match target resolution. "
            f"Expected {(latent_h, latent_w)}, got {tuple(init_image_latents.shape[-2:])}."
        )

    if mask_image is None:
        return init_image_latents, None

    mask_tensor = prepare_inpaint_mask_tensor(
        mask_image,
        width=width,
        height=height,
    )
    mask_latents = F.interpolate(
        mask_tensor,
        size=(latent_h, latent_w),
        mode="nearest",
    )
    inpaint_mask = mask_latents.to(
        device=pipe.execution_device, dtype=sample_dtype
    ).unsqueeze(2)
    inpaint_mask = inpaint_mask.repeat(1, init_image_latents.shape[1], 1, 1, 1)
    inpaint_mask = align_tensor_batch_size(
        inpaint_mask,
        target_batch_size=batch_size,
        input_name="mask_image",
    )
    return init_image_latents, inpaint_mask


def _generate_image(
    pipe: "AnimaPipeline",
    *,
    prompt: PromptInput,
    negative_prompt: PromptInput | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    image: ImageBatchInput | None = None,
    mask_image: ImageBatchInput | None = None,
    strength: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 32,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 4.0,
    generator: GeneratorInput | None = None,
    sampler: str = "euler_a_rf",
    sigma_schedule: str = "beta",
    beta_alpha: float = FORGE_BETA_ALPHA,
    beta_beta: float = FORGE_BETA_BETA,
    eta: float = 1.0,
    s_noise: float = 1.0,
    er_sde_max_stage: int = 3,
    cfg_batch_mode: str = "auto",
    sample_dtype: str | torch.dtype = "auto",
    check_finite: bool = False,
    output_type: str = "pil",
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
) -> list[Image.Image] | torch.Tensor:
    """Internal end-to-end generation routine used by ``AnimaPipeline.__call__``."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    use_cfg = guidance_scale > 1.0
    effective_cfg_batch_mode = _resolve_effective_cfg_batch_mode(
        cfg_batch_mode, execution_device=pipe.execution_device
    )
    if prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0]
        pos_hidden = pos_qwen_mask = pos_t5_ids = pos_t5_mask = pos_t5_weights = None
        neg_hidden = neg_qwen_mask = neg_t5_ids = neg_t5_mask = neg_t5_weights = None
    else:
        prompts, negative_prompts = _resolve_prompt_batches(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        batch_size = len(prompts)
        (
            pos_hidden, pos_qwen_mask, pos_t5_ids, pos_t5_mask, pos_t5_weights,
            neg_hidden, neg_qwen_mask, neg_t5_ids, neg_t5_mask, neg_t5_weights,
        ) = _prepare_prompt_embedding_inputs(
            pipe,
            prompt=prompts,
            negative_prompt=negative_prompts if use_cfg else None,
        )

    height, width, latent_h, latent_w = latent_hw(
        height=height,
        width=width,
        vae_scale_factor=pipe.vae_scale_factor,
        patch_size=pipe.patch_size,
    )
    resolved_sample_dtype = _resolve_sample_dtype(
        sample_dtype,
        model_dtype=pipe.model_dtype,
        execution_device=pipe.execution_device,
    )

    resolved_callback_tensor_inputs = callback_on_step_end_tensor_inputs or ["latents"]
    init_generator, step_generator, noise_device, noise_dtype = _resolve_noise_runtime(
        execution_device=pipe.execution_device,
        generator=generator,
        batch_size=batch_size,
    )

    init_image_latents: torch.Tensor | None = None
    inpaint_mask: torch.Tensor | None = None
    init_noise: torch.Tensor | None = None

    init_image_latents, inpaint_mask = _prepare_init_image_latents_and_inpaint_mask(
        pipe,
        image=image,
        mask_image=mask_image,
        width=width,
        height=height,
        latent_h=latent_h,
        latent_w=latent_w,
        batch_size=batch_size,
        init_generator=init_generator,
        sample_dtype=resolved_sample_dtype,
    )

    flowmatch_timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    input_is_noisy_latents = False

    if sampler == "flowmatch_euler":
        flowmatch_timesteps = _trim_flowmatch_timesteps_by_strength(
            pipe,
            num_inference_steps=num_inference_steps,
            strength=strength if init_image_latents is not None else 1.0,
        )
        if init_image_latents is None:
            latents = randn_tensor(
                (batch_size, 16, 1, latent_h, latent_w),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            )
            latents = latents.to(device=pipe.execution_device, dtype=resolved_sample_dtype)
        else:
            init_image_latents = align_tensor_batch_size(
                init_image_latents,
                target_batch_size=batch_size,
                input_name="image",
            )
            init_noise = randn_tensor(
                tuple(init_image_latents.shape),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            ).to(device=pipe.execution_device, dtype=resolved_sample_dtype)
            start_timestep = (
                flowmatch_timesteps[:1]
                .expand(init_image_latents.shape[0])
                .to(
                    device=pipe.execution_device,
                    dtype=torch.float32,
                )
            )
            latents = pipe.scheduler.scale_noise(
                init_image_latents, start_timestep, init_noise
            )
            latents = latents.to(device=pipe.execution_device, dtype=resolved_sample_dtype)
    else:
        sigmas = build_sampling_sigmas(
            pipe.scheduler,
            num_inference_steps=num_inference_steps,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            device=pipe.execution_device,
        )
        if init_image_latents is None:
            latents = randn_tensor(
                (batch_size, 16, 1, latent_h, latent_w),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            )
            latents = latents.to(device=pipe.execution_device, dtype=resolved_sample_dtype)
        else:
            init_image_latents = align_tensor_batch_size(
                init_image_latents,
                target_batch_size=batch_size,
                input_name="image",
            )
            sigmas = _trim_sigmas_by_strength(
                sigmas=sigmas,
                strength=strength,
            )
            init_noise = randn_tensor(
                tuple(init_image_latents.shape),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            ).to(device=pipe.execution_device, dtype=resolved_sample_dtype)
            sigma_start = sigmas[0].to(init_image_latents.dtype)
            latents = (
                sigma_start * init_noise + (1.0 - sigma_start) * init_image_latents
            )
            input_is_noisy_latents = True

    with _module_execution_context(
        pipe.transformer,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        if prompt_embeds is not None:
            pos_cond = prompt_embeds.to(
                device=pipe.execution_device, dtype=pipe.model_dtype
            )
            if use_cfg:
                neg_cond = negative_prompt_embeds.to(  # type: ignore[union-attr]
                    device=pipe.execution_device, dtype=pipe.model_dtype
                )
            else:
                neg_cond = None
        else:
            pos_cond, neg_cond = _build_cfg_conditions_from_embeddings(
                pipe,
                pos_hidden=pos_hidden,
                pos_qwen_mask=pos_qwen_mask,
                pos_t5_ids=pos_t5_ids,
                pos_t5_mask=pos_t5_mask,
                pos_t5_weights=pos_t5_weights,
                neg_hidden=neg_hidden,
                neg_qwen_mask=neg_qwen_mask,
                neg_t5_ids=neg_t5_ids,
                neg_t5_mask=neg_t5_mask,
                neg_t5_weights=neg_t5_weights,
            )

        pos_cond = pos_cond.to(device=pipe.execution_device, dtype=pipe.model_dtype)
        if neg_cond is not None:
            neg_cond = neg_cond.to(device=pipe.execution_device, dtype=pipe.model_dtype)
            if effective_cfg_batch_mode == "concat":
                # Precompute the static CFG conditioning batch once. The previous
                # path rebuilt this concatenation at every denoising step.
                pos_cond = torch.cat([pos_cond, neg_cond], dim=0)
                neg_cond = None

        if sampler == "flowmatch_euler":
            if flowmatch_timesteps is None:
                raise RuntimeError(
                    "Internal error: flowmatch timesteps were not initialized."
                )
            latents = sample_flowmatch_euler(
                pipe.transformer,
                pipe.scheduler,
                pipe,
                latents,
                timesteps=flowmatch_timesteps,
                sigma_schedule=sigma_schedule,
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                guidance_scale=guidance_scale,
                cfg_batch_mode=effective_cfg_batch_mode,
                model_dtype=pipe.model_dtype,
                check_finite=check_finite,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )
        else:
            if sigmas is None:
                raise RuntimeError(
                    "Internal error: sigma schedule was not initialized."
                )
            latents = run_const_sigma_samplers(
                pipe.transformer,
                pipe,
                latents,
                sigmas=sigmas,
                sampler=sampler,
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                guidance_scale=guidance_scale,
                eta=eta,
                s_noise=s_noise,
                generator=step_generator,
                cfg_batch_mode=effective_cfg_batch_mode,
                model_dtype=pipe.model_dtype,
                check_finite=check_finite,
                er_sde_max_stage=er_sde_max_stage,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                input_is_noisy_latents=input_is_noisy_latents,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )

    if output_type == "latent":
        return latents

    with _module_execution_context(
        pipe.vae,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        return decode_latents(pipe.vae, latents, runtime_dtype=pipe.model_dtype)


# ---------------------------------------------------------------------------
# AnimaPipeline
# ---------------------------------------------------------------------------


class AnimaPipeline(DiffusionPipeline, AnimaLoraLoaderMixin):
    """Diffusers pipeline wrapper for Anima text-to-image, img2img, and inpaint.

    Anima defaults to ``AnimaFlowMatchEulerDiscreteScheduler``. You can override
    the scheduler component by passing ``scheduler=...`` to ``from_pretrained``.
    """

    transformer: AnimaTransformerModel
    vae: AutoencoderKLQwenImage
    scheduler: AnimaFlowMatchEulerDiscreteScheduler
    text_encoder: PreTrainedModel
    prompt_tokenizer: AnimaPromptTokenizer | None
    execution_device: str
    model_dtype: torch.dtype
    text_encoder_dtype: torch.dtype
    use_module_cpu_offload: bool
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    # prompt_tokenizer is intentionally NOT registered via register_modules because
    # AnimaPromptTokenizer is a custom class without Diffusers save_pretrained/from_pretrained
    # support. It is stored as a plain instance attribute and its constituent tokenizers are
    # saved/loaded via the overridden save_pretrained / from_pretrained methods.
    _optional_components: list[str] = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        *,
        transformer: AnimaTransformerModel,
        vae: AutoencoderKLQwenImage,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel,
        prompt_tokenizer: AnimaPromptTokenizer | None = None,
        execution_device: str = "auto",
        model_dtype: torch.dtype = torch.float32,
        text_encoder_dtype: torch.dtype = torch.float32,
        use_module_cpu_offload: bool = False,
    ):
        super().__init__()
        resolved_scheduler = coerce_anima_scheduler(scheduler)
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=resolved_scheduler,
            text_encoder=text_encoder,
        )

        # Diffusers passes [None, None] for model_index.json entries with null library/class.
        # Normalize to None so downstream checks work correctly.
        if isinstance(prompt_tokenizer, (list, tuple)):
            prompt_tokenizer = None
        self.prompt_tokenizer = prompt_tokenizer
        self.execution_device = execution_device
        self.model_dtype = model_dtype
        self.text_encoder_dtype = text_encoder_dtype
        self.use_module_cpu_offload = use_module_cpu_offload
        # Runtime-only representation bridge. It is intentionally not registered
        # as a Diffusers module or persisted with the pipeline.
        self.text_encoder_bridge: AnimaTextEncoderBridge | None = None
        self.vae_scale_factor = resolve_vae_scale_factor(vae=self.vae)
        self.patch_size = resolve_patch_size(transformer=self.transformer)

    def load_text_encoder_bridge(
        self,
        path: str | Path,
        *,
        center_strength: float = 0.5,
        variance_strength: float = 0.0,
    ) -> "AnimaPipeline":
        """Load a calibration bridge mapping the active Qwen encoder to Anima's Qwen3-0.6B space."""
        bridge = AnimaTextEncoderBridge.from_file(
            path,
            center_strength=center_strength,
            variance_strength=variance_strength,
        )
        active_family = _active_text_encoder_family(self.text_encoder)
        bridge_source = str(bridge.metadata.get("source_family", "unknown"))
        if bridge_source not in {"", "unknown", active_family}:
            raise ValueError(
                "Text-encoder bridge source family does not match the active encoder: "
                f"bridge={bridge_source!r}, active={active_family!r}."
            )
        hidden_size = int(getattr(getattr(self.text_encoder, "config", None), "hidden_size", 0) or 0)
        if hidden_size and hidden_size != bridge.hidden_size:
            raise ValueError(
                f"Text-encoder bridge hidden size {bridge.hidden_size} does not match active encoder {hidden_size}."
            )
        self.text_encoder_bridge = bridge
        return self

    def set_text_encoder_bridge(
        self, bridge: AnimaTextEncoderBridge | None
    ) -> "AnimaPipeline":
        self.text_encoder_bridge = bridge
        return self

    def clear_text_encoder_bridge(self) -> "AnimaPipeline":
        if self.text_encoder_bridge is not None:
            self.text_encoder_bridge.clear_runtime_cache()
        self.text_encoder_bridge = None
        return self

    def set_qwen_source_max_length(self, max_length: int | None) -> "AnimaPipeline":
        """Optionally cap only the Qwen source-memory length; None keeps it unrestricted by Anima."""
        if self.prompt_tokenizer is None:
            raise RuntimeError("prompt_tokenizer is not initialised")
        if max_length is not None and int(max_length) < 1:
            raise ValueError("max_length must be >= 1 or None")
        self.prompt_tokenizer.qwen_source_max_length = (
            None if max_length is None else int(max_length)
        )
        return self

    def set_t5_query_strategy(self, strategy: str) -> "AnimaPipeline":
        """Choose how the bounded T5/query side covers an overlength prompt.

        ``"uniform"`` (default) samples query tokens across the complete text,
        while ``"head"`` reproduces ordinary first-511-token truncation. The
        Qwen source memory is independent and is never shortened by this option.
        """
        if self.prompt_tokenizer is None:
            raise RuntimeError("prompt_tokenizer is not initialised")
        normalized = str(strategy).strip().lower()
        if normalized not in {"head", "uniform"}:
            raise ValueError("strategy must be 'head' or 'uniform'")
        self.prompt_tokenizer.t5_query_strategy = normalized
        return self

    def set_adapter_source_position_mode(self, mode: str) -> "AnimaPipeline":
        """Control adapter-side RoPE positions for Qwen source memories longer than 512.

        ``"compress"`` keeps every source KV token but maps its adapter RoPE
        coordinate into the original 0..511 range. ``"raw"`` uses monotonically
        increasing positions beyond 511 for research / A-B comparison.
        """
        normalized = str(mode).strip().lower()
        if normalized not in {"compress", "raw"}:
            raise ValueError("mode must be 'compress' or 'raw'")
        self.transformer.llm_adapter.source_position_mode = normalized
        return self

    @property
    def execution_device(self) -> str:
        override = getattr(self, "_anima_execution_device", None)
        if isinstance(override, str) and override != "auto":
            return override
        # CPU-offload path: Diffusers sets _execution_device to the inference GPU.
        device = getattr(self, "_execution_device", None)
        if device is not None:
            return device.type
        # "auto" or unset: detect from the transformer's current device so that
        # pipe.to("cuda") is reflected without an explicit execution_device assignment.
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            try:
                return next(transformer.parameters()).device.type
            except StopIteration:
                pass
        return "cpu"

    @execution_device.setter
    def execution_device(self, value: str) -> None:
        self._anima_execution_device = str(value)

    @property
    def model_dtype(self) -> torch.dtype:
        override = getattr(self, "_anima_model_dtype", None)
        if isinstance(override, torch.dtype):
            return override
        dtype = getattr(self, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.float32

    @model_dtype.setter
    def model_dtype(self, value: torch.dtype) -> None:
        self._anima_model_dtype = value

    @property
    def spatial_step(self) -> int:
        """Return the required pixel step for width/height alignment."""
        return self.vae_scale_factor * self.patch_size

    def check_inputs(
        self,
        *,
        prompt: PromptInput,
        negative_prompt: PromptInput | None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image: ImageBatchInput | None,
        mask_image: ImageBatchInput | None,
        strength: float,
        width: int,
        height: int,
        num_inference_steps: int,
        num_images_per_prompt: int,
        generator: GeneratorInput | None,
        sampler: str,
        sigma_schedule: str,
        cfg_batch_mode: str,
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> None:
        """Validate user-facing call arguments and runtime constraints."""
        if (prompt_embeds is None) != (negative_prompt_embeds is None):
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must both be provided "
                "or both be None."
            )
        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            prompts, _ = _resolve_prompt_batches(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
            )
            batch_size = len(prompts)

        if strength <= 0.0 or strength > 1.0:
            raise ValueError("`strength` must be in (0.0, 1.0].")
        step = self.spatial_step
        if width < step or height < step:
            raise ValueError(f"`width` and `height` must be >= {step}.")
        if width % step != 0 or height % step != 0:
            suggested_height = height - (height % step)
            suggested_width = width - (width % step)
            raise ValueError(
                f"`width` and `height` must be divisible by {step} but are {width} and {height}. "
                f"Try width={suggested_width}, height={suggested_height}."
            )
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be >= 1.")
        if image is None and mask_image is not None:
            raise ValueError("`mask_image` requires `image`.")
        if image is None and not math.isclose(strength, 1.0):
            raise ValueError("`strength` can be changed only when `image` is provided.")

        _validate_image_like_input(image, input_name="image")
        _validate_image_like_input(mask_image, input_name="mask_image")
        _normalize_generator(generator, batch_size=batch_size)
        _validate_sampling_modes(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            cfg_batch_mode=cfg_batch_mode,
            output_type=output_type,
        )
        _validate_callback_tensor_input_names(
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            allowed_inputs=self._callback_tensor_inputs,
        )

    @torch.inference_mode()
    def encode_prompt_plan(
        self,
        prompt_plan: Any | list[Any],
        negative_prompt_plan: Any | list[Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sd_embed-style structured plans without building/mixing separate conditions.

        Each plan contains one full visible text plus character-span factors.
        AND/BREAK are metadata/group boundaries inside that single memory rather
        than requests to create multiple completed conditioning tensors.
        """
        pos_plans = prompt_plan if isinstance(prompt_plan, list) else [prompt_plan]
        if negative_prompt_plan is None:
            neg_plans = [{"text": "", "spans": []} for _ in pos_plans]
        else:
            neg_plans = (
                negative_prompt_plan
                if isinstance(negative_prompt_plan, list)
                else [negative_prompt_plan]
            )
            if len(neg_plans) == 1 and len(pos_plans) > 1:
                neg_plans = neg_plans * len(pos_plans)
            if len(neg_plans) != len(pos_plans):
                raise ValueError("positive/negative prompt-plan batch sizes must match")
        pos_cond = _encode_prompt_plan_batch(self, list(pos_plans))
        neg_cond = _encode_prompt_plan_batch(self, list(neg_plans))
        return pos_cond, neg_cond

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: PromptInput,
        negative_prompt: PromptInput | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to conditioning tensors.

        Pre-computing conditioning with this method allows reusing the same
        embeddings across multiple ``__call__`` invocations (e.g. sweeping seeds),
        avoiding redundant tokenisation and text-encoding on each call.

        Example::

            pos_cond, neg_cond = pipe.encode_prompt(prompt, negative_prompt)
            for seed in seeds:
                pipe(prompt, negative_prompt,
                     prompt_embeds=pos_cond, negative_prompt_embeds=neg_cond,
                     generator=torch.Generator().manual_seed(seed))

        Args:
            prompt: Text prompt(s) for generation.
            negative_prompt: Optional negative prompt(s). Defaults to empty string.
            num_images_per_prompt: Number of images to generate per prompt entry.

        Returns:
            A tuple ``(pos_cond, neg_cond)`` where each tensor has shape
            ``(batch_size, 512, text_embed_dim)``. Pass these as
            ``prompt_embeds`` / ``negative_prompt_embeds`` to ``__call__``.
        """
        prompts, negative_prompts = _resolve_prompt_batches(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        (
            pos_hidden, pos_qwen_mask, pos_t5_ids, pos_t5_mask, pos_t5_weights,
            neg_hidden, neg_qwen_mask, neg_t5_ids, neg_t5_mask, neg_t5_weights,
        ) = _prepare_prompt_embedding_inputs(
            self,
            prompt=prompts,
            negative_prompt=negative_prompts,
        )
        with _module_execution_context(
            self.transformer,
            execution_device=self.execution_device,
            execution_dtype=self.model_dtype,
            enable_offload=self.use_module_cpu_offload,
        ):
            pos_cond, neg_cond = _build_cfg_conditions_from_embeddings(
                self,
                pos_hidden=pos_hidden,
                pos_qwen_mask=pos_qwen_mask,
                pos_t5_ids=pos_t5_ids,
                pos_t5_mask=pos_t5_mask,
                pos_t5_weights=pos_t5_weights,
                neg_hidden=neg_hidden,
                neg_qwen_mask=neg_qwen_mask,
                neg_t5_ids=neg_t5_ids,
                neg_t5_mask=neg_t5_mask,
                neg_t5_weights=neg_t5_weights,
            )
        return pos_cond, neg_cond

    @torch.inference_mode()
    def __call__(
        self,
        prompt: PromptInput,
        negative_prompt: PromptInput | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image: ImageBatchInput | None = None,
        mask_image: ImageBatchInput | None = None,
        strength: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 32,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 4.0,
        generator: GeneratorInput | None = None,
        cfg_batch_mode: str = "auto",
        sample_dtype: str | torch.dtype = "auto",
        check_finite: bool = False,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[..., dict[str, Any] | None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> AnimaPipelineOutput | tuple[list[Image.Image] | np.ndarray | torch.Tensor]:
        """Generate images with Anima.

        Args:
            prompt: Text prompt(s) for generation.
            negative_prompt: Optional negative prompt(s).
            prompt_embeds: Pre-computed positive conditioning tensor from
                ``encode_prompt``. When provided, ``prompt`` is ignored for text
                encoding and this tensor is used directly. Requires
                ``negative_prompt_embeds`` to be provided as well.
            negative_prompt_embeds: Pre-computed negative conditioning tensor from
                ``encode_prompt``. Must be provided together with ``prompt_embeds``.
            image: Optional initial image for img2img or inpainting.
            mask_image: Optional inpaint mask (white = region to inpaint).
            strength: Noise strength for img2img (0.0–1.0].
            width: Output image width (must be divisible by ``spatial_step``).
            height: Output image height (must be divisible by ``spatial_step``).
            num_inference_steps: Number of denoising steps.
            num_images_per_prompt: Number of images to generate per prompt.
            guidance_scale: Classifier-free guidance scale.
            generator: Optional RNG seed(s).
            cfg_batch_mode: How to run classifier-free guidance. ``auto`` uses
                Diffusers-style batched CFG on CUDA and ``split`` elsewhere.
                ``split`` runs positive and negative conditioning as two sequential
                forward passes. ``concat`` batches them into a single forward.
            sample_dtype: Latent/sampling dtype. ``auto`` uses the model dtype on
                CUDA/MPS when possible and float32 on CPU. Use ``float32`` to keep
                the older more conservative path.
            check_finite: Run per-step NaN/Inf checks for fp16 debugging. Disabled
                by default because it forces GPU synchronisation every step.
            output_type: ``pil``, ``np``, or ``latent``.
            return_dict: Return ``AnimaPipelineOutput`` when ``True``.
            callback_on_step_end: Optional callable invoked at each step end.
            callback_on_step_end_tensor_inputs: Tensor names passed to the callback.

        Notes:
            Sampling parameters (sampler, sigma_schedule, eta, s_noise, beta_alpha,
            beta_beta, er_sde_max_stage) are first-class scheduler config options. Set them with
            ``pipe.scheduler.set_sampling_config(...)`` before calling.

            - ``flowmatch_euler`` requires ``sigma_schedule='uniform'``.
            - ``eta`` and ``s_noise`` are ignored for ``flowmatch_euler`` and ``euler``.
            - ``eta`` is ignored for ``er_sde``; ``s_noise`` controls ER-SDE stochasticity.
            - ``er_sde_max_stage`` is used only for ``er_sde`` and must be 1, 2, or 3.
            - ``beta_alpha`` and ``beta_beta`` are used only for ``sigma_schedule='beta'``.
        """
        sampling_config: AnimaSamplingConfig = self.scheduler.get_sampling_config()
        sampler = sampling_config.sampler
        sigma_schedule = sampling_config.sigma_schedule
        beta_alpha = sampling_config.beta_alpha
        beta_beta = sampling_config.beta_beta
        eta = sampling_config.eta
        s_noise = sampling_config.s_noise
        er_sde_max_stage = sampling_config.er_sde_max_stage
        resolved_callback_tensor_inputs = callback_on_step_end_tensor_inputs
        if resolved_callback_tensor_inputs is None:
            resolved_callback_tensor_inputs = ["latents"]

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask_image,
            strength=strength,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            cfg_batch_mode=cfg_batch_mode,
            output_type=output_type,
            callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
        )
        _warn_ignored_sampling_arguments(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            eta=eta,
            s_noise=s_noise,
        )

        try:
            images = _generate_image(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=image,
                mask_image=mask_image,
                strength=strength,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                generator=generator,
                sampler=sampler,
                sigma_schedule=sigma_schedule,
                beta_alpha=beta_alpha,
                beta_beta=beta_beta,
                eta=eta,
                s_noise=s_noise,
                er_sde_max_stage=er_sde_max_stage,
                cfg_batch_mode=cfg_batch_mode,
                sample_dtype=sample_dtype,
                check_finite=check_finite,
                output_type=output_type,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
            )
        finally:
            self.maybe_free_model_hooks()

        if output_type == "pil":
            output_images: list[Image.Image] | np.ndarray | torch.Tensor = images
        elif output_type == "np":
            output_images = np.stack(
                [np.asarray(image, dtype=np.uint8) for image in images], axis=0
            )
        else:
            output_images = images

        if not return_dict:
            return (output_images,)
        return AnimaPipelineOutput(images=output_images)

    def compile_components(
        self,
        *,
        transformer: bool = True,
        vae: bool = False,
        mode: str | None = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool | None = None,
        backend: str | None = None,
    ) -> "AnimaPipeline":
        """Compile hot inference modules with ``torch.compile`` and return ``self``.

        This mirrors the common Diffusers optimisation pattern where users compile
        the denoiser/transformer after loading. Compilation has a first-call cost,
        so it is intended for repeated generation with the same pipeline.
        """
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            raise RuntimeError("torch.compile is not available in this PyTorch build.")

        kwargs: dict[str, Any] = {"fullgraph": fullgraph}
        if mode is not None:
            kwargs["mode"] = mode
        if dynamic is not None:
            kwargs["dynamic"] = dynamic
        if backend is not None:
            kwargs["backend"] = backend

        if transformer:
            self.transformer = compile_fn(self.transformer, **kwargs)
        if vae:
            self.vae = compile_fn(self.vae, **kwargs)
        return self

    def enable_torch_compile(
        self,
        **kwargs: Any,
    ) -> "AnimaPipeline":
        """Backward-friendly alias for ``compile_components``."""
        return self.compile_components(**kwargs)

    def enable_model_cpu_offload(
        self,
        gpu_id: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Move models between CPU and GPU around each inference stage.

        Replaces the Diffusers hook-based offload with Anima's
        ``_module_execution_context`` mechanism, which is aware of the
        manual text-encoder → transformer → VAE execution order used by this
        pipeline (including the adapter conditioning pass that runs before
        the main denoising loop).

        Args:
            gpu_id: Deprecated; ignored. Configure the execution device via
                the ``execution_device`` attribute instead.
            device: The accelerator device to use. Accepts a device string
                (``"cuda"``, ``"cuda:1"``, ``"mps"``) or ``torch.device``.
                Ignored when ``execution_device`` was set explicitly.
        """
        if getattr(self, "_anima_execution_device", "auto") == "auto":
            self._anima_execution_device = device.type if isinstance(device, torch.device) else str(device)
        self.use_module_cpu_offload = True

    def enable_vae_slicing(self) -> None:
        """Enable VAE slicing when the backend VAE implementation supports it."""
        _enable_vae_method(
            self.vae,
            enabled=True,
            method_name="enable_slicing",
            unsupported_feature_name="VAE slicing",
        )

    def disable_vae_slicing(self) -> None:
        """Disable VAE slicing when the backend VAE implementation supports it."""
        _disable_vae_method(
            self.vae,
            method_name="disable_slicing",
            unsupported_feature_name="VAE slicing",
        )

    def enable_vae_tiling(self) -> None:
        """Enable VAE tiling when the backend VAE implementation supports it."""
        _enable_vae_method(
            self.vae,
            enabled=True,
            method_name="enable_tiling",
            unsupported_feature_name="VAE tiling",
        )

    def disable_vae_tiling(self) -> None:
        """Disable VAE tiling when the backend VAE implementation supports it."""
        _disable_vae_method(
            self.vae,
            method_name="disable_tiling",
            unsupported_feature_name="VAE tiling",
        )

    def enable_vae_xformers_memory_efficient_attention(self) -> None:
        """Enable xformers memory-efficient attention for the VAE when supported."""
        method = getattr(self.vae, "set_use_memory_efficient_attention_xformers", None)
        if method is None:
            warnings.warn(
                "VAE xformers is not supported by the current VAE implementation.",
                UserWarning,
                stacklevel=2,
            )
            return
        try:
            method(True)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Failed to enable VAE xformers attention: {exc}",
                stacklevel=2,
            )

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Save the pipeline and bundled prompt tokenizer artifacts."""
        super().save_pretrained(save_directory, **kwargs)
        if self.prompt_tokenizer is None:
            return
        save_prompt_tokenizers_to_local_dir(
            prompt_tokenizer=self.prompt_tokenizer,
            save_directory=Path(save_directory),
        )

    @classmethod
    def clear_component_cache(cls) -> None:
        """Clear the raw-file loader cache used by from_single_file/from_multiple_files."""
        clear_anima_component_cache()

    @classmethod
    def _from_pretrained_local_directory(
        cls,
        pretrained_model_name_or_path: str,
        *,
        pipeline_dir: Path,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        load_options = loader_options_from_kwargs(kwargs, consume=False)
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        normalize_loaded_component_buffers(loaded)
        loaded.scheduler = coerce_anima_scheduler(loaded.scheduler)

        if loaded.prompt_tokenizer is None:
            qwen_source, t5_source, uses_local_tokenizers = (
                resolve_prompt_tokenizer_sources_for_local_dir(
                    pipeline_dir=pipeline_dir,
                )
            )
            tokenizer_options = load_options
            if uses_local_tokenizers:
                tokenizer_options = AnimaLoaderOptions(
                    local_files_only=True,
                    cache_dir=load_options.cache_dir,
                    force_download=False,
                    token=load_options.token,
                    revision=load_options.revision,
                    proxies=load_options.proxies,
                    cache_components=load_options.cache_components,
                    cache_transformer=False,
                )
            loaded.prompt_tokenizer = load_prompt_tokenizer(
                qwen_tokenizer_source=qwen_source,
                t5_tokenizer_source=t5_source,
                options=tokenizer_options,
            )
        return loaded

    @classmethod
    def _from_pretrained_diffusers_repo(
        cls,
        pretrained_model_name_or_path: str,
        *,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        load_options = loader_options_from_kwargs(kwargs, consume=False)
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        normalize_loaded_component_buffers(loaded)
        loaded.scheduler = coerce_anima_scheduler(loaded.scheduler)

        if loaded.prompt_tokenizer is None:
            # Tokenizers are always bundled in the Diffusers-format repository under
            # fixed subdirectory names. If not present locally, loading.py falls back
            # to the fixed Anima HF repository sources.
            from .loading import _QWEN_TOKENIZER_SOURCE, _T5_TOKENIZER_SOURCE

            loaded.prompt_tokenizer = load_prompt_tokenizer(
                qwen_tokenizer_source=_QWEN_TOKENIZER_SOURCE,
                t5_tokenizer_source=_T5_TOKENIZER_SOURCE,
                options=load_options,
            )
        return loaded

    @classmethod
    def _from_components_source(
        cls,
        components: AnimaComponents,
        *,
        kwargs: dict[str, Any],
        api_name: str,
    ) -> "AnimaPipeline":
        ignored, unknown = _partition_single_file_from_pretrained_kwargs(kwargs)
        for key in ignored:
            kwargs.pop(key, None)
        if ignored:
            warnings.warn(
                f"Ignoring unsupported {api_name} arguments for Anima single-file loading: "
                + ", ".join(ignored),
                stacklevel=2,
            )

        scheduler = scheduler_from_kwargs(kwargs, consume=False)
        runtime_options = runtime_options_from_kwargs(kwargs, consume=False)
        load_options = loader_options_from_kwargs(kwargs, consume=False)
        if unknown:
            raise ValueError(
                f"Unsupported arguments for AnimaPipeline.{api_name}: {', '.join(unknown)}"
            )

        runtime = build_anima_pipeline(
            components=components,
            device=runtime_options.device,
            dtype=runtime_options.dtype,
            text_encoder_dtype=runtime_options.text_encoder_dtype,
            local_files_only=load_options.local_files_only,
            cache_dir=load_options.cache_dir,
            force_download=load_options.force_download,
            token=load_options.token,
            revision=load_options.revision,
            proxies=load_options.proxies,
            scheduler=scheduler,
        )
        return runtime

    @classmethod
    def _from_single_file_source(
        cls,
        pretrained_model_name_or_path: str,
        *,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        return cls._from_components_source(
            AnimaComponents(model_path=str(pretrained_model_name_or_path)),
            kwargs=kwargs,
            api_name="from_single_file",
        )

    @classmethod
    def _from_multiple_file_sources(
        cls,
        model_path: str,
        encoder_path: str,
        vae_path: str,
        *,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        return cls._from_components_source(
            AnimaComponents(
                model_path=str(model_path),
                text_encoder_path=str(encoder_path),
                vae_path=str(vae_path),
            ),
            kwargs=kwargs,
            api_name="from_multiple_files",
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> "AnimaPipeline":
        """Load Anima from a Diffusers-format pipeline directory or Hub repository."""
        _raise_if_removed_from_pretrained_runtime_feature_kwargs(
            kwargs, api_name="from_pretrained"
        )
        custom_single_file_only = sorted(
            key for key in kwargs if key in _ANIMA_COMPONENT_OVERRIDE_KEYS
        )
        custom_runtime_only = sorted(
            key for key in kwargs if key in {"device", "dtype", "text_encoder_dtype"}
        )
        unsupported_custom = custom_single_file_only + custom_runtime_only
        if unsupported_custom:
            raise ValueError(
                "Unsupported `from_pretrained` arguments for Diffusers-format loading: "
                + ", ".join(unsupported_custom)
                + ". Use standard Diffusers `from_pretrained(...)` kwargs for converted repositories "
                + "or `from_single_file(...)` for raw checkpoints."
            )
        scheduler = kwargs.get("scheduler", None)
        if scheduler is not None:
            kwargs["scheduler"] = coerce_anima_scheduler(scheduler)

        source = str(pretrained_model_name_or_path)
        if _looks_like_single_file_source(source):
            raise ValueError(
                "`from_pretrained` only supports Diffusers-format repositories/directories. "
                "Use `AnimaPipeline.from_single_file(...)` for raw `.safetensors` checkpoints."
            )

        path = Path(source)
        if path.is_dir() and (path / "model_index.json").exists():
            return cls._from_pretrained_local_directory(
                pretrained_model_name_or_path,
                pipeline_dir=path,
                kwargs=kwargs,
            )
        return cls._from_pretrained_diffusers_repo(
            pretrained_model_name_or_path,
            kwargs=kwargs,
        )

    @classmethod
    def from_single_file(
        cls, pretrained_model_link_or_path: str, **kwargs: Any
    ) -> "AnimaPipeline":
        """Load Anima from a single-file checkpoint with Diffusers-like kwargs."""
        _raise_if_removed_from_pretrained_runtime_feature_kwargs(
            kwargs, api_name="from_single_file"
        )
        _pop_ignored_kwargs(
            kwargs,
            ignored_keys=_DIFFUSERS_COMPAT_IGNORED_FROM_SINGLE_FILE_KEYS,
            api_name="from_single_file",
        )
        return cls._from_single_file_source(
            pretrained_model_link_or_path,
            kwargs=kwargs,
        )

    @classmethod
    def from_multiple_files(
        cls,
        model_path: str,
        encoder_path: str,
        vae_path: str,
        **kwargs: Any,
    ) -> "AnimaPipeline":
        """Load Anima from separate transformer, text encoder, and VAE files.

        ``model_path`` points to the Anima transformer checkpoint,
        ``encoder_path`` points to a supported Qwen text encoder checkpoint
        (Qwen3-0.6B or Qwen3.5-0.8B text backbone), and
        ``vae_path`` points to the Anima VAE checkpoint. Each path accepts the
        same source forms as ``from_single_file``: a local file,
        ``repo_id::filename``, or a Hugging Face file URL.

        Reload speed options:
        - ``cache_components=True`` keeps text encoder, VAE, and tokenizers in an
          in-process cache.
        - ``cache_transformer=True`` also caches the transformer when the exact
          same model file is reloaded. This is fastest but keeps the model in
          memory until ``AnimaPipeline.clear_component_cache()`` is called.
        """
        _raise_if_removed_from_pretrained_runtime_feature_kwargs(
            kwargs, api_name="from_multiple_files"
        )
        _pop_ignored_kwargs(
            kwargs,
            ignored_keys=_DIFFUSERS_COMPAT_IGNORED_FROM_SINGLE_FILE_KEYS,
            api_name="from_multiple_files",
        )
        return cls._from_multiple_file_sources(
            model_path,
            encoder_path,
            vae_path,
            kwargs=kwargs,
        )
