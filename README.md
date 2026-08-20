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

## TCAtria1B text encoder (Team-C)

This fork can use **TCAtria1B** as the Anima text encoder without changing the
Anima / Anima 2.9B transformer checkpoint format.

A TCAtria1B directory is detected when it contains:

```text
TCAtria1B/
├── hybrid_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json              # optional when tokenizer.json is complete
```

Use that directory as `encoder_path`:

```python
pipe = AnimaPipeline.from_multiple_files(
    model_path="/workspace/anima_or_anima29b.safetensors",
    encoder_path="/workspace/anima_hybrid_1b_understanding_work/02_understanding",
    vae_path="/workspace/qwen_image_vae.safetensors",
    torch_dtype=torch.bfloat16,
    text_encoder_dtype="bf16",
    text_encoder_max_sequence_length=1024,
)
```

The same call supports both 28-block Anima and 40-block Anima 2.9B because the
transformer depth is inferred independently from the text encoder. TCAtria1B
emits 1024-wide hidden states, so the existing Anima LLM adapter remains
unchanged.

### Context handling

For TCAtria1B, the loader defaults the source text encoder to 1024 tokens (the
current calibration length). The T5 target side and final Anima conditioning
remain 512 tokens. This lets the existing LLM adapter compress a longer
TCAtria source sequence into Anima's native 512-token conditioning.

Stock Qwen3-0.6B loading is unchanged and defaults to 512 source tokens.
