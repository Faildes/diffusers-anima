# Anima text encoder profile v2

`anima_text_encoder_profile_v2` is a self-describing safetensors format for mapping an alternate text encoder into the representation space expected by Anima's Qwen3-0.6B-trained LLM adapter.

## Artifact kinds

- `bridge_profile`: bridge tensors only. Load the source encoder normally, then attach the profile with `pipe.load_text_encoder_bridge(path)`.
- `aligned_encoder`: bridge tensors plus `encoder.*` weights. Pass the file directly as `encoder_path`; the pipeline loads the embedded encoder and automatically attaches the bridge.

## Tensor namespace

Required:

- `bridge.rotation`
- `bridge.source_mean`
- `bridge.target_mean`

Optional:

- `bridge.variance_scale`
- `bridge.rms_scale`
- `bridge.input_projection` — reserved for future source encoders whose hidden width differs from the Anima reference width.
- `encoder.*` — source text encoder state dict when `artifact_kind=aligned_encoder`.

## Transform semantics

v2 defines two endpoints and interpolates between them with `center_strength`:

- `0.0`: pure rotated source representation, `x @ R`.
- `1.0`: centered Procrustes representation, `(x - source_mean) @ R + target_mean`.

Variance calibration is applied around the centered representation and is controlled separately by `variance_strength`.

The calibration script evaluates center/variance combinations on a held-out split and stores the best values as `recommended_center_strength` and `recommended_variance_strength`. Omitting strengths in `load_text_encoder_bridge()` uses those stored recommendations.

## Calibration corpus

`--prompts` is optional. Without it, the script deterministically creates a 4096-line visual corpus containing atomic tags, tag-style prompts, spatial relations, camera/composition, lighting/materials, and long prompts. Phrase expansion creates many more paired anchors.

For domain-specific models, supplying an additional representative prompt corpus is still recommended. Keep the calibration corpus about visual semantics rather than image-quality outcomes; the goal is representation alignment, not fine-tuning.

## Long prompt policy

The Qwen source sequence and Anima target/query sequence are separate:

- Qwen source memory may exceed 512 positions.
- T5 target/query remains at the native Anima 512-position contract.
- `uniform` T5 query selection samples anchors across the full prompt instead of dropping the tail.
- adapter source RoPE can be compressed into the learned range while keeping all source KV tokens.

## JupyterLab calibration

`calibrate_text_encoder_bridge.py` is importable as a Python module. For JupyterLab, use `BridgeCalibrationConfig`, `preview_calibration_corpus()`, `make_jupyter_progress_callback()`, and `calibrate_text_encoder_bridge()` instead of shelling out to the CLI. A ready-to-run notebook is included at `notebooks/Calibrate_Anima_Text_Encoder_Bridge.ipynb`.

Calibration-corpus construction rules are documented in `docs/calibration_prompt_corpus.md`.
