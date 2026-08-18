# diffusers-anima

> **Anima 2.9B compatibility:** raw single-file loading infers the main transformer depth from checkpoint keys. Original 28-block Anima and expanded 40-block Anima 2.9B checkpoints are both supported, including `net.*` and `model.diffusion_model.*` wrapper prefixes.


`diffusers-anima` provides an Anima pipeline implementation designed to align with [Diffusers](https://github.com/huggingface/diffusers) patterns.

## Install

```bash
pip install git+https://github.com/hdae/diffusers-anima.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/hdae/diffusers-anima.git
```

## Quick Start

### Text-to-image

```python
import torch
from diffusers_anima import AnimaPipeline

pipe = AnimaPipeline.from_pretrained(
    "hdae/diffusers-anima-preview",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
result.images[0].save("output.png")
```

### From a single-file checkpoint

```python
pipe = AnimaPipeline.from_single_file("/path/to/anima.safetensors")
```

### Img2Img / Inpaint

```python
from PIL import Image

init_image = Image.open("input.png").convert("RGB")
mask_image = Image.open("mask.png").convert("L")  # white = repaint area

# Img2Img:
result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    image=init_image,
    strength=0.65,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
)

# Inpaint:
result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    image=init_image,
    mask_image=mask_image,
    strength=0.75,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
)
```

### LoRA

```python
pipe.load_lora_weights("/path/to/lora.safetensors", adapter_name="style")
pipe.set_adapters("style", adapter_weights=[0.8])

result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    width=1024, 
    height=1024,
)
```

## Model

Model weights are hosted on HuggingFace: [hdae/diffusers-anima-preview](https://huggingface.co/hdae/diffusers-anima-preview)

The Anima model was developed by [CircleStone Labs](https://huggingface.co/circlestone-labs) and [Comfy Org](https://github.com/Comfy-Org).
The weights are based on [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) and are subject to the
[CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md).
**The model weights may not be used for commercial purposes.** Generated outputs may be used for any purpose.

This code library (`diffusers-anima`) is separately licensed under Apache 2.0.

## Documentation

| Document | Description |
|---|---|
| [`docs/api.md`](docs/api.md) | Full API reference (loading, generation, sampling config) |
| [`docs/development.md`](docs/development.md) | Development setup, test commands, project structure |
| [`docs/custom_implementations.md`](docs/custom_implementations.md) | Intentional deviations from Diffusers upstream |
| [`docs/text_encoder_profiles.md`](docs/text_encoder_profiles.md) | Qwen3.5/alternate encoder compatibility profile format |
| [`docs/native_text_encoder.md`](docs/native_text_encoder.md) | Bridge-free Anima-native Qwen3.5-0.8B training and runtime format |
| [`docs/final_encoder_v4_stability.md`](docs/final_encoder_v4_stability.md) | v3 final encoder + v4 saturation/binding stability layer |

## Long prompts beyond 512 tokens

v5 keeps Anima's trained 512-slot target conditioning contract but removes the
512-token **source-memory cliff** inside the frozen LLM adapter. Long Qwen
memories are handled by overlapping native-size attention banks plus a
per-query bank router, instead of one >512-key softmax with compressed RoPE.
See [`docs/long_context_v5.md`](docs/long_context_v5.md).

## Experimental aligned text-encoder profiles

Anima's LLM adapter was trained against Qwen3-0.6B hidden states. Alternate
encoders such as Qwen3.5-0.8B are shape-compatible at 1024 dimensions but are
not representation-compatible by default. This repository therefore supports a
self-describing `anima_text_encoder_profile_v2` artifact.

The calibration corpus is optional on the command line; a deterministic 4096-line
visual corpus is built in. You can also supply your own corpus with `--prompts`.

```bash
python scripts/calibrate_text_encoder_bridge.py \
  --source-model /path/to/qwen3.5-0.8b-base.safetensors \
  --reference-model /path/to/qwen3-0.6b-base.safetensors \
  --output /path/to/qwen35_08b_to_qwen3_06b.profile.safetensors \
  --dump-prompts /path/to/prompts_used.txt
```

The small profile is loaded next to an external encoder:

```python
pipe = AnimaPipeline.from_multiple_files(
    model_path=anima_path,
    encoder_path=qwen35_path,
    vae_path=vae_path,
)
pipe.load_text_encoder_bridge(
    "/path/to/qwen35_08b_to_qwen3_06b.profile.safetensors"
)  # uses held-out recommended strengths from the profile
```

For a future single-file aligned encoder, create the same profile with source
weights bundled into the `encoder.*` namespace:

```bash
python scripts/calibrate_text_encoder_bridge.py \
  --source-model /path/to/qwen3.5-0.8b-base.safetensors \
  --reference-model /path/to/qwen3-0.6b-base.safetensors \
  --output /path/to/anima_qwen35_08b_aligned.safetensors \
  --bundle-source-weights
```

That output can replace the ordinary encoder path directly:

```python
pipe = AnimaPipeline.from_multiple_files(
    model_path=anima_path,
    encoder_path="/path/to/anima_qwen35_08b_aligned.safetensors",
    vae_path=vae_path,
)
# The embedded bridge is detected and attached automatically.
```

In both modes, Qwen source-memory length is separate from Anima's 512 target/query
contract. `set_qwen_source_max_length(None)` keeps the source unrestricted by
Anima, `set_t5_query_strategy("uniform")` spreads bounded T5 query anchors across
the full prompt, and `set_adapter_source_position_mode("compress")` keeps long
source KV while normalizing adapter RoPE positions into the learned range.

The v2 profile stores architecture fingerprints, calibration-corpus hash,
held-out validation metrics, recommended center/variance strengths, and optional
future `bridge.input_projection` support for wider encoders. Calibration uses no
gradients or optimizer.



## Anima-native Qwen3.5-0.8B encoder

For the bridge-free final path, train an `anima_native_text_encoder_v1` checkpoint with
`scripts/train_anima_native_text_encoder.py`.  It retains the Qwen3.5-0.8B backbone,
adds an integrated trainable multi-layer Anima head, and uses the Qwen3-0.6B encoder
only as a compatibility reference during training.  The resulting single safetensors
file is passed directly as `encoder_path`; no runtime bridge or v3 conditioner is
attached. See [`docs/native_text_encoder.md`](docs/native_text_encoder.md).

## Final Anima text encoder v3

For a performance-first single-file encoder, convert a calibrated v2 profile with
`scripts/finalize_anima_text_encoder.py`. The resulting `anima_text_encoder_v3`
artifact keeps the complete source backbone and embeds one global conditioning
head. Its aligned token memory remains the primary signal; optional bounded
semantic summary slots preserve more Qwen3.5 source geometry as additional KV
memory. See [`docs/final_text_encoder.md`](docs/final_text_encoder.md).
