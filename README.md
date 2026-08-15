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

## Anima 2.9B + Qwen3.5 semantic frontend

Raw three-file loading can use the original Qwen3-0.6B encoder or an official
Qwen3.5-0.8B checkpoint. The loader detects Qwen3.5 from its hybrid
`linear_attn` / `model.language_model` keys and automatically selects the
Qwen3.5 tokenizer. For an official full Qwen3.5 checkpoint, the loader extracts
only `model.language_model.*` into the official text-only causal-LM wrapper.
This keeps `generate()` available for the optional prompt compiler without
allocating Qwen3.5's unused vision tower or MTP modules, so one Qwen checkpoint
still serves both compilation and Anima text encoding.

```python
from diffusers_anima import AnimaPipeline

pipe = AnimaPipeline.from_multiple_files(
    "Anima-2.9B.safetensors",
    "Qwen3.5-0.8B.safetensors",
    "vae.safetensors",
    dtype="bfloat16",
)
```

`AnimaPipeline.set_prompt_processor(processor)` installs an inference-only
frontend. A processor may implement `process_batch(prompts, negative=False)` or
be a callable. The image model still receives exactly 512 conditioning
positions; long-prompt understanding/compression happens before tokenisation for
the Anima LLM adapter.
