# Anima final text encoder v3

`anima_text_encoder_v3` is the performance-first successor to the external v2 bridge.
It is a **single encoder file** containing the complete source Qwen backbone plus one
global Anima conditioning head.

The v3 design deliberately does not rewrite/distil the 0.8B backbone. The goal is
not to replace Qwen3.5's representation with Qwen3-0.6B, but to keep the wider
understanding of Qwen3.5 while presenting Anima with a compatible primary memory.

## Memory layout

At inference:

1. Qwen3.5-0.8B produces its normal full-length hidden states.
2. `head.*` produces the **primary aligned memory** expected by Anima.
3. Optionally, the same single global transform produces a covariance-preserving
   source view. A small number of segment-pooled residual slots are appended as
   *extra* source KV memory.
4. The existing Anima LLM adapter cross-attends the combined memory. T5/query
   remains <=512 exactly as before.

The expansion path never replaces aligned tokens. It is bounded by a residual-norm
clip and is therefore suitable for conservative A/B tuning.

## File layout

```text
format = anima_text_encoder_v3
artifact_kind = final_encoder
anima_ready = true

encoder.*
head.rotation
head.source_mean
head.target_mean
head.variance_scale        # optional
head.rms_scale             # optional
head.input_projection      # future wider source encoders
```

There are no language-specific or task-specific bridges.

## Convert a v2 bridge/profile

If the v2 profile already contains `encoder.*`:

```bash
python scripts/finalize_anima_text_encoder.py \
  --bridge-profile /workspace/qwen35_aligned.profile.safetensors \
  --output /workspace/Anima-Qwen3.5-0.8B-Final.safetensors
```

If it is a small bridge-only profile:

```bash
python scripts/finalize_anima_text_encoder.py \
  --bridge-profile /workspace/qwen35_08b_to_qwen3_06b.profile.safetensors \
  --source-model /workspace/qwen3.5-0.8b-base.safetensors \
  --output /workspace/Anima-Qwen3.5-0.8B-Final.safetensors
```

Then use the result directly as `encoder_path`. The pipeline detects and attaches
the embedded final conditioning head automatically.

## Performance-first defaults

```text
semantic_expansion_strength          0.25
semantic_expansion_max_tokens        16
semantic_expansion_chunk_size        64
semantic_expansion_min_source_tokens 16
semantic_expansion_residual_clip     0.35
```

For strict compatibility A/B, use `pipe.set_semantic_expansion(strength=0.0)`.
For capability-retention tests compare 0.15 / 0.25 / 0.35 at fixed prompt+seed.
The primary aligned memory is unchanged across these tests.

## Why this is not bridge distillation

Absorbing the bridge into Qwen weights can reduce the source model's useful
representation geometry. v3 keeps the backbone untouched and makes Anima
compatibility an encoder-local output contract. A later native distillation path
can still be researched, but v3 is intended to be the quality-preserving final
format unless image-level evaluation proves destructive absorption is better.
