# Anima 2.9B quality profile

Anima 2.9B is not just a wider checkpoint for the original runtime. Its main
transformer has 40 blocks instead of 28: 28 inherited base blocks are
interleaved with 12 newly trained expansion blocks. Reusing every base-model
runtime assumption can therefore produce a valid image while silently reducing
quality.

## Automatic behaviour

| Area | Base Anima (28 blocks) | Anima 2.9B (40 blocks) |
|---|---|---|
| Sampler default | `euler_a_rf` + `beta` | `euler` + `uniform` |
| Auto CFG execution | concat on CUDA, split elsewhere | split on every device |
| Auto latent dtype | float32 | float32 |
| Complete base LoRA | native indices | mapped to inherited 2.9B indices |

The sampler profile follows the released model's recommended Euler +
SGM-uniform path. The latent remains float32 even if transformer weights use
bfloat16, matching the published 2.9B reference pipeline. Transformer inference
still runs at the configured model dtype.

```python
pipe = AnimaPipeline.from_single_file(
    "/path/to/anima-2.9b.safetensors",
    torch_dtype=torch.bfloat16,
)

# Architecture is detected from the checkpoint.
assert pipe.model_variant == "2.9b"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=832,
    height=1216,
    num_inference_steps=28,
    guidance_scale=4.0,
    # quality-safe architecture-aware defaults:
    cfg_batch_mode="auto",
    sample_dtype="auto",
).images[0]
```

For a high-quality run, the model card recommends roughly 28–50 steps and CFG
3.5–5. Explicit `bfloat16` or `float16` latent sampling and concatenated CFG
remain available, but are performance/VRAM tradeoffs rather than the default
quality path.

## LoRA migration

Block numbers after the first insertion no longer identify the same semantic
layer. The loader detects a complete LoRA containing all base blocks `0..27`
and remaps it to the inherited 2.9B block positions.

For a partial LoRA whose keys are entirely within `0..27`, choose the intended
layout explicitly:

```python
# LoRA trained for original 28-block Anima
pipe.load_lora_weights(path, anima_lora_layout="base28")

# LoRA trained directly for Anima 2.9B
pipe.load_lora_weights(path, anima_lora_layout="native")
```

This ambiguity is rejected instead of silently targeting the wrong 40-block
layers.

## sd_embed / Artist Mixer

Use the matching revised `sd_embed` package in this bundle. It maps Artist
Mixer interventions to the 28 inherited layers and leaves all newly trained
expansion blocks untouched. Its `long_prompt_strategy="auto"` also keeps Anima
2.9B conditioning at the native 512-token length using residual chunk fusion.
Explicit `chunk_concat` is still supported for comparisons.

References: [Anima 2.9B model card](https://huggingface.co/Gazingstars123/Anima-2.9B),
[published reference pipeline](https://huggingface.co/spaces/akhaliq/Anima-2.9B/blob/main/pipeline.py).
