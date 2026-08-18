# Anima-native Qwen3.5-0.8B text encoder

`anima_native_text_encoder_v1` is the bridge-free endpoint for Qwen3.5-0.8B use with Anima.

The final runtime path is:

```text
Qwen3.5-0.8B backbone
  -> token-dependent multi-layer mixer
  -> Anima semantic residual block
  -> learned 1024->1024 Anima projection
  -> Anima LLM Adapter
```

The native checkpoint contains both `encoder.*` and `native.*` weights.  It does **not** contain a runtime `bridge.*` transform and does not attach `AnimaTextEncoderBridge` or the v3 `AnimaTextEncoderConditioner` when loaded.

## Why this differs from the bridge path

A bridge has one dominant objective: map the alternate encoder into the representation space learned from Qwen3-0.6B.  That is useful for compatibility, but it can also collapse distinctions that Qwen3.5-0.8B represents better.

Native training therefore uses Qwen3-0.6B only as an **Anima-compatibility reference**.  Qwen3.5 remains the knowledge/geometry reference.  The loss keeps source concept distances and token geometry while relaxing the 0.6B-reference term on minimal pairs for which the 0.8B source separates concepts more strongly.

## 40k calibration prompt file

The trainer accepts the existing UTF-8 one-prompt-per-line calibration file directly.  Blank lines and `#` comments are ignored and duplicate lines are removed.

The training corpus helper adds deterministic constraints on top of the original lines:

- color / appearance minimal pairs
- count pairs
- left/right, front/behind and camera pairs
- sitting/standing and other pose pairs
- saturation / palette pairs
- generated two-subject binding pairs where only one subject changes

The original prompt text is never replaced by these generated constraints; they are additional groups.

## Recommended staged training

### Stage 1 — native head only

Keep the complete Qwen3.5 backbone frozen.  Train only the integrated native head.

```bash
python scripts/train_anima_native_text_encoder.py \
  --source-model /workspace/qwen35_08b_base.safetensors \
  --reference-model /workspace/qwen3_06b_base.safetensors \
  --prompts /workspace/calibration_prompts.txt \
  --bootstrap-bridge-profile /workspace/qwen35_to_qwen3.profile.safetensors \
  --train-last-n-layers 0 \
  --output /workspace/anima_qwen35_native_stage1.safetensors
```

The historical bridge is optional and is used only to initialise/supervise Stage 1.  No bridge is required to load the output.

### Stage 2 — last four Qwen3.5 layers

```bash
python scripts/train_anima_native_text_encoder.py \
  --resume-native /workspace/anima_qwen35_native_stage1.safetensors \
  --reference-model /workspace/qwen3_06b_base.safetensors \
  --prompts /workspace/calibration_prompts.txt \
  --train-last-n-layers 4 \
  --head-lr 5e-5 \
  --backbone-lr 3e-6 \
  --output /workspace/anima_qwen35_native_stage2.safetensors
```

### Stage 3 — last eight layers, only if Stage 2 improves image tests

Use a lower backbone LR.  The 40k prompt set is appropriate for controlled adaptation, but not for unconstrained full-0.8B fine-tuning.

```bash
python scripts/train_anima_native_text_encoder.py \
  --resume-native /workspace/anima_qwen35_native_stage2.safetensors \
  --reference-model /workspace/qwen3_06b_base.safetensors \
  --prompts /workspace/calibration_prompts.txt \
  --train-last-n-layers 8 \
  --head-lr 2e-5 \
  --backbone-lr 1e-6 \
  --output /workspace/anima_qwen35_native_final.safetensors
```

## Loss design

The trainer uses the following terms:

- `anima_compat`: pooled cosine + normalised MSE against Qwen3-0.6B.
- `source_geometry`: pairwise prompt geometry from Qwen3.5-0.8B.
- `token_geometry`: within-prompt token relation geometry, preserving subject/attribute separation without requiring identical absolute coordinate systems.
- `knowledge_gain`: for generated minimal pairs, preserve the 0.8B distance when it exceeds the 0.6B distance; simultaneously relax the 0.6B compatibility term for that pair.
- `distribution`: keep token RMS statistics inside an Anima-compatible range.
- `layer_prior`: very small prior that allows useful upper-middle Qwen layers to contribute instead of forcing final-layer-only conditioning.
- `bootstrap_token`: optional, decaying token-level teacher loss from the old stable bridge.

The bridge bootstrap is intentionally decayed.  It is an initial Anima coordinate hint, not the final target.

## Runtime

Use the native file directly as `encoder_path`:

```python
pipe = AnimaPipeline.from_multiple_files(
    model_path=anima_path,
    encoder_path="/workspace/anima_qwen35_native_final.safetensors",
    vae_path=vae_path,
)

print(pipe.describe_text_encoder_profile())
# native_encoder=True
# anima_ready=True
# bridge_required=False
```

No call to `load_text_encoder_bridge()` or `load_text_encoder_conditioner()` should be made for a native encoder; the pipeline rejects those combinations to prevent double alignment.

With sd_embed, final validation can enforce native-only runtime:

```python
embeds, neg = get_weighted_text_embeddings_anima(
    pipe,
    prompt=prompt,
    neg_prompt=negative_prompt,
    use_prompt_plan=True,
    require_aligned_text_encoder=True,
    require_native_text_encoder=True,
)
```

## Artifact tensor namespace

```text
encoder.*   trained Qwen3.5 text backbone
native.*    Anima-native layer mixer / semantic block / projection
```

Metadata includes:

```text
format=anima_native_text_encoder_v1
artifact_kind=native_encoder
contains_encoder_weights=true
anima_ready=true
bridge_required=false
runtime_conditioning_mode=native_encoder_direct
```

The tokenizer remains the Qwen3.5 tokenizer.  Anima's T5 query side and existing long-source/windowed Adapter path remain unchanged.
