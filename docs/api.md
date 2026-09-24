# API Reference

## Loading

### `AnimaPipeline.from_pretrained`

```python
AnimaPipeline.from_pretrained(
    pretrained_model_name_or_path: str,
    **kwargs,
) -> AnimaPipeline
```

Loads from a Diffusers-format pipeline directory or Hugging Face Hub repository.
The repository must contain a `model_index.json` file.

Accepts standard Diffusers `from_pretrained` kwargs such as `cache_dir`,
`token`, `revision`, `local_files_only`, and `scheduler`.

> **Note:** Runtime options such as `device`, `dtype`, `text_encoder_dtype`,
> and VAE/offload toggles are **not** accepted here. Configure them after
> loading with the standard pipeline methods.

### `AnimaPipeline.from_single_file`

```python
AnimaPipeline.from_single_file(
    pretrained_model_link_or_path: str,
    *,
    # Component sources (defaults to hdae/diffusers-anima-preview)
    text_encoder_weights: str = ...,
    text_encoder_config_repo: str = ...,
    qwen_tokenizer_repo: str = ...,
    t5_tokenizer_repo: str = ...,
    vae_repo: str = ...,
    # Runtime
    device: str = "auto",
    dtype: str = "auto",
    text_encoder_dtype: str = "auto",
    # Loader options
    local_files_only: bool = False,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
) -> AnimaPipeline
```

Loads from a raw `.safetensors` checkpoint. The transformer is loaded from the
given path; all other components default to the Anima preview repository on
Hugging Face.

**Path formats accepted for `pretrained_model_link_or_path`:**
- Local file: `"/path/to/anima.safetensors"`
- HF URL: `"https://huggingface.co/owner/repo/blob/main/model.safetensors"`
- Repo+filename: `"owner/repo::path/to/model.safetensors"`

The same `repo::filename` shorthand applies to all component source args.

---

## Generation — `AnimaPipeline.__call__`

```python
pipe(
    prompt: str | list[str],
    negative_prompt: str | list[str] | None = None,
    image: ImageInput | None = None,
    mask_image: ImageInput | None = None,
    strength: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 32,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 4.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    cfg_batch_mode: str = "auto",
    sample_dtype: str | torch.dtype = "auto",
    output_type: str = "pil",
    return_dict: bool = True,
    callback_on_step_end: Callable | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
) -> AnimaPipelineOutput
```

### Key parameters

| Parameter | Description |
|---|---|
| `prompt` | Positional argument. Single string or list of strings. |
| `image` | PIL Image / ndarray / tensor for img2img or inpainting. |
| `mask_image` | Inpaint mask: white pixels are inpainted, black are preserved. |
| `strength` | How much noise to add for img2img (0.0–1.0]. 1.0 = full generation. |
| `cfg_batch_mode` | `"auto"`, `"split"`, or `"concat"`. Auto uses split CFG for 2.9B and the historical CUDA concat path for base Anima. |
| `sample_dtype` | `"auto"` keeps denoising latents and CFG arithmetic in float32. Explicit bf16/fp16 is a lower-memory quality tradeoff. |
| `output_type` | `"pil"` (default), `"np"` (uint8 array), or `"latent"`. |

### Width / Height constraints

`width` and `height` must be divisible by `pipe.spatial_step`
(= `vae_scale_factor × patch_size`, typically **16**).

---

## Sampling Configuration

Sampling parameters are stored in the scheduler config and are serialised with
`save_pretrained`. Change them with:

```python
pipe.scheduler.set_sampling_config(
    sampler="euler_a_rf",      # flowmatch_euler | euler | euler_a_rf | euler_ancestral_rf
    sigma_schedule="beta",     # uniform | beta | simple | normal
    beta_alpha=0.5,
    beta_beta=0.5,
    eta=1.0,
    s_noise=1.0,
)
```

`from_single_file` and the pipeline constructor detect a 40-block Anima 2.9B
transformer. When its scheduler still has the historical base-model defaults,
they are upgraded to `sampler="euler"` and `sigma_schedule="uniform"`. Explicit
non-default scheduler configurations are preserved. To reapply the detected
recommendation later, call `pipe.use_recommended_sampling_config()`.

### Sampler/schedule matrix

| `sampler` | Compatible `sigma_schedule` | Notes |
|---|---|---|
| `flowmatch_euler` | `uniform` only | Uses Diffusers scheduler step |
| `euler` | `beta`, `simple`, `normal` | Classic Euler; `eta`/`s_noise` ignored |
| `euler_a_rf` | `beta`, `simple`, `normal` | Ancestral RF Euler |
| `euler_ancestral_rf` | `beta`, `simple`, `normal` | Alias for `euler_a_rf` |

---

## Runtime Helpers

Choose one runtime path after loading the pipeline.

### Balanced (existing)

```python
pipe.enable_persistent_transformer_staging("cuda:0")
pipe.enable_fast_denoising(mode="balanced")
```

This keeps the Transformer on CUDA, stages Qwen/VAE, batches CFG, and compiles
the repeated blocks. For less VRAM, use `cfg_batch_mode="split"`.

### Maximum (new)

Use a fresh pipeline with LoRA and transformer patches configured, then:

```python
pipe.enable_fast_denoising(mode="maximum", device="cuda:0")
```

All components remain on CUDA; CFG is batched, and the repeated transformer
blocks use `default` compilation with CUDA Graphs explicitly disabled. This
reuses compiled code for the 28/40 blocks without risking a previously produced
Cosmos block tensor being overwritten by CUDA Graph replay. Do not call the
balanced compiler or `compile_components()` on the same pipeline first.
The default path passes CUDA Graph settings via `options` only; PyTorch versions
that reject specifying both `mode` and `options` are supported.
Benchmark repeated calls with unchanged tensor shapes; regional compilation
still takes time on the first call. The separate first-forward tqdm uses an
indeterminate counter because compilation has no measurable percentage. CPU
generators keep their existing seed behavior.
Model weights, scheduler, denoising steps, and latent dtype stay unchanged;
compiled kernels can have small floating-point differences.
With an explicit CPU generator and `euler_a_rf`, maximum mode also prepares the
same ancestral noise draws before sampling and transfers them to CUDA together.
This reduces repeated CPU-to-GPU transfers without changing the generator's
final state. Prefetch is skipped for callbacks, non-CPU generators, and noise
stacks over 512 MiB. Pass `prefetch_cpu_noise=False` to disable it for a speed
comparison or memory-constrained run.

For more aggressive kernel tuning without CUDA Graph replay, pass
`compile_mode="max-autotune-no-cudagraphs"` on a fresh pipeline; initial
compilation can take longer. The CUDA Graph modes `reduce-overhead` and
`max-autotune` are rejected because they can overwrite outputs between Cosmos
blocks. If compilation is unavailable, use `compile_blocks=False` to retain
GPU residency and CFG batching. To benchmark full-model compilation, pass
`compile_scope="full"` on a fresh pipeline; it may compile considerably longer.

### Maximum throughput with precomputed embeddings

Maximum mode keeps models resident, but it cannot merge consecutive calls to
`pipe(...)` by itself. When generating several images from the same SD_Embed
embeddings, send 2–4 seeds in one call to use otherwise idle VRAM and increase
images per second:

```python
pipe.enable_fast_denoising(mode="maximum", device="cuda:0")
pipe.enable_dev_metrics(sample_interval=0.1)  # optional benchmark output
batch_size = 2  # try 2, then 4 if VRAM permits
for offset in range(0, num_gen, batch_size):
    batch_seeds = [int(s) for s in seeds[offset : offset + batch_size]]
    images = pipe(
        prompt=None,
        prompt_embeds=embeds,
        negative_prompt_embeds=negative_embeds,
        width=w, height=h,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=len(batch_seeds),
        generator=[torch.Generator("cpu").manual_seed(s) for s in batch_seeds],
    ).images
    for index, image in enumerate(images, start=offset):
        seed = int(seeds[index])
        # Move the existing geninfo, PngInfo, display and image.save block here.
        # Keep image.save(..., pnginfo=metadata) for each index.
```

`num_images_per_prompt` now repeats each precomputed positive/negative embed
row for that many images, and validates that the generator list matches the
expanded batch. Each independent CPU generator advances in the same order as
the corresponding single-image call. The model sees a larger batch, so fused
kernels may produce small numerical differences. Image size, model precision,
sampler, step count, guidance, and RNG algorithm are unchanged. Measure warm
calls using `pipe.dev_metrics_last["seconds_per_image"]` and its `vram` peaks;
compare both speed and output on your GPU. Batch generation improves throughput
when generating multiple images; it does not shorten a single-image request.
For a single-image benchmark, you can test `compile_mode="max-autotune-no-cudagraphs"`
on a freshly loaded pipeline; its first compile can take considerably longer.

For VAE slicing/tiling or manual CPU offload, select them separately when
memory is the priority:

```python
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()
```

### Dev metrics (optional)

```python
pipe.enable_dev_metrics(sample_interval=0.1)  # compatible with either mode
image = pipe(prompt_embeds=embeds, negative_prompt_embeds=negative_embeds, ...).images[0]
report = pipe.dev_metrics_last          # dict, also printed after each call
pipe.disable_dev_metrics()              # optional; leaves the last report intact
# pipe.reset_dev_metrics()              # restart the cumulative time and call count
```

The report gives call duration, accumulated call time, average time per call,
and, on CUDA, VRAM at the start, time-weighted average, peak, and end. It shows
both PyTorch allocated and reserved memory; reserved includes the caching
allocator. `GPU used (all processes)` shows device-wide VRAM when available and
may include other workloads. Average and device-wide peak are sampled at the
configured interval; the PyTorch allocated/reserved peaks come from the CUDA
allocator's exact peak counters, reset at the start of each monitored call.
The call duration includes first-forward compilation, denoising, and decoding;
it excludes embedding work done before `pipe(...)` and saving/displaying the
result afterward. When a call returns a batch, time and VRAM apply to the whole
batch. CPU/MPS runs still report time and mark VRAM unavailable. This optional
feature synchronizes CUDA at call boundaries and polls memory during execution,
so disable it for production throughput measurements.

---

## LoRA

```python
pipe.load_lora_weights("owner/repo", weight_name="lora.safetensors")
pipe.set_adapters(["lora_name"], adapter_weights=[0.8])
pipe.fuse_lora()
pipe.unfuse_lora()
pipe.unload_lora_weights()
```

On a 40-block model, a complete legacy 28-block LoRA is mapped to its 28
inherited layers automatically. A partial LoRA whose keys all lie in `0..27` is
ambiguous and intentionally raises; pass `anima_lora_layout="base28"` to remap
it or `anima_lora_layout="native"` for a native 40-block LoRA.

See `AnimaLoraLoaderMixin` for the full API surface.
