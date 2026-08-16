# Calibration prompt corpus rules

This document defines the recommended input corpus for `scripts/calibrate_text_encoder_bridge.py`.
The corpus is **not training data**. It is used only to measure paired hidden representations from the source encoder and the Anima reference encoder, then solve a representation-space alignment transform.

## File format

- UTF-8 plain text.
- One complete prompt per line.
- Empty lines are ignored.
- Lines whose first non-space character is `#` are comments and are ignored.
- Commas and semicolons are meaningful because phrase expansion splits each line on `,` and `;` and also keeps the full line as an anchor.
- Keep one semantic prompt on one physical line. Do not split one prompt over several lines in the text file.

Example:

```text
1girl, short black hair, blue eyes, white jacket, leaning toward viewer, low angle, rainy city street, rim light
2women; left subject with red hair; right subject with black hair; facing each other; medium shot; warm cafe interior
android girl, polished chrome limbs, transparent visor, reaching toward camera, strong foreshortening, mechanical hangar
```

## Corpus size

- Hard minimum accepted by the calibrator: **256 prompt lines**.
- Minimum expanded anchors: **512**.
- Practical experimental size: **1024+ lines**.
- Recommended general-purpose baseline: **4096+ lines**.
- Larger corpora are useful only when they add semantic coverage. Repeating near-identical prompts does not substitute for variety.

With phrase expansion enabled (default), every full prompt is retained and comma/semicolon-delimited phrases are also added as unique anchors. Therefore 4096 prompt lines normally produce far more than 4096 paired calibration anchors.

## What the corpus should represent

The source and reference encoders should be compared on the concepts that Anima actually needs to consume. A general-purpose corpus should cover all of the following rather than concentrating on one style or one character type:

1. **Subject/count/identity** — solo, duo, multiple subjects, people, creatures, robots, roles.
2. **Appearance** — hair, eyes, face, body shape, age presentation, visible attributes.
3. **Clothing/accessories** — garment types, layering, materials, jewelry, props.
4. **Pose/action** — standing, sitting, crouching, reaching, twisting, running, interacting.
5. **Spatial relations** — left/right, foreground/background, behind/in front of, facing, overlap, contact.
6. **Camera** — close-up, cowboy shot, full body, low/high angle, dutch angle, lens-like perspective wording.
7. **Composition** — thirds, diagonals, symmetry/asymmetry, focal hierarchy, negative space, group arrangement.
8. **Environment** — interiors, landscapes, city scenes, mechanical/fantasy spaces, weather.
9. **Lighting/color** — daylight, rim light, backlight, neon, volumetric light, palette relationships.
10. **Materials/rendering** — metal, glass, fabric, wood, gloss, linework, cel shading, painterly rendering.
11. **Text/interface concepts** — signs, logos, readable text, blank signs, UI panels, when relevant to your use.
12. **Long coherent prompts** — combinations of the above in one consistent scene, to calibrate contextual drift rather than only isolated tags.

## Recommended length distribution

Do not build the corpus entirely from long prompts. The bridge needs both local concept directions and global/contextual directions.

A good general-purpose mixture is approximately:

- **20–30% atomic or very short anchors**: one concept or a few tightly related concepts.
- **25–35% short prompts**: roughly 2–8 comma/semicolon clauses.
- **25–35% medium prompts**: roughly 9–16 clauses.
- **15–25% long prompts**: 17+ coherent clauses, including the kinds of long prompts you actually intend to use.

The built-in corpus follows this same principle: atomic concepts first, then short tag-style prompts, medium relation/material prompts, and finally long coherent prompts.

## Coherence rule for long prompts

A calibration line should describe **one interpretable scene**. Long examples should add detail without introducing alternative mutually exclusive compositions.

Good:

```text
1girl, red hair, black jacket, running toward viewer, one hand in foreground, low angle, strong foreshortening, rainy night street, blue-orange lighting
```

Avoid:

```text
1girl, sitting, standing, front view, back view, close-up, full body, indoor classroom, desert landscape
```

The second line does not provide a clean semantic direction; it combines mutually conflicting scene states.

## Frontend syntax rule

Calibration should represent **semantic text**, not sd_embed/A1111 control syntax. Prefer plain phrases.

Avoid making these a large fraction of the corpus:

- `(red hair:1.5)` / `[red hair:0.8]`
- `BREAK`
- top-level `AND`
- Artist Mixer syntax
- prompt scheduling syntax

These controls belong to the prompt parser/runtime layer. The encoder bridge should align the hidden representation of the underlying concepts. A small number of real-world syntax-bearing prompts is acceptable for robustness, but they should not dominate the corpus.

## Phrase expansion rule

By default the calibrator stores the full line **and** unique pieces split on commas/semicolons. This is intentional:

```text
1girl, red hair, blue eyes, low angle, forest
```

contributes anchors for the full scene plus local concepts such as `red hair`, `blue eyes`, `low angle`, and `forest`.

For that reason:

- Use commas/semicolons at meaningful semantic boundaries.
- Do not create huge comma-free paragraphs if you want strong phrase-level alignment.
- Do not over-fragment one concept into unnatural single words.
- Repeated identical phrases are deduplicated case-insensitively, so repetition does not increase their calibration weight.

## Character, artist, and domain-specific names

The default corpus intentionally does not depend on a named-character/artist list. For a domain-specific encoder profile, adding representative names can be useful, but keep them a minority of the corpus and preserve the general visual categories above.

A practical rule is:

- Keep the general-purpose visual corpus as the majority.
- Add a domain supplement containing the characters, styles, technical terminology, languages, or specialized concepts you actually use.
- Do not make one franchise, artist, or style dominate unless the resulting bridge is intentionally domain-specific.

## Languages

Match the language distribution to real use. If prompts are almost always English/Danbooru-style tags, keep the corpus mostly English. If the bridge must support Japanese or other languages, include representative prompts in those languages rather than assuming an English-only calibration transfers perfectly.

Do not mix languages randomly within every line unless that reflects actual prompting behavior.

## Negative-prompt vocabulary

Negative prompts use the same text encoder, so a small amount of common negative vocabulary can be included when it is important to your workflow, for example `blurry`, `extra limbs`, `watermark`, or `duplicate subject`.

Do not turn the corpus into a quality-score dataset. Calibration is aligning **representation directions**, not learning which image result is good or bad.

## Long-context coverage and `max_length`

The calibrator tokenizes every anchor with `max_length` (default 2048). Anything beyond this limit is truncated for that calibration pass.

If the intended runtime uses long prompts:

- Include a meaningful long-prompt tail in the corpus.
- Set calibration `max_length` high enough to cover those examples within available VRAM.
- Prefer many coherent long examples with different compositions over a few extremely huge prompts.
- Keep short/atomic anchors as well; long-only calibration weakens local concept alignment.

## Duplicate and balance rules

Avoid:

- Copying the same prompt many times.
- Hundreds of lines differing only by one color when other categories are missing.
- A corpus dominated by `1girl, solo` while multi-subject/spatial relations are absent.
- A corpus dominated by one camera angle or one style.

The Jupyter helper `preview_calibration_corpus()` reports duplicate counts, approximate length buckets, frontend-syntax frequency, and warnings before loading either encoder.

## Holdout validation

The script shuffles expanded anchors with a deterministic seed and keeps a validation split (`validation_fraction`, default 0.08). Do not manually duplicate validation lines in the source corpus. The same corpus can be reproduced through the saved seed and SHA-256 metadata.

When comparing bridge versions, keep the corpus, seed, pooling mode, `max_length`, and validation fraction fixed so cosine/RMSE changes remain meaningful.

## Recommended baseline settings

For Qwen3.5-0.8B-Base -> Qwen3-0.6B-Base:

```python
BridgeCalibrationConfig(
    default_prompt_count=4096,
    batch_size=8,          # increase if VRAM allows
    pooling="both",
    last_weight=0.65,
    max_length=2048,
    include_phrases=True,
    min_phrase_chars=3,
    validation_fraction=0.08,
    seed=3571,
)
```

The `last_weight` value only affects `pooling="blend"`; `pooling="both"` emits both mean-pooled and causal-endpoint anchors.

## Suggested workflow for a custom corpus

1. Start with the built-in 4096-line corpus and produce a baseline bridge.
2. Create a custom supplement following the rules above.
3. Combine general + custom prompts rather than replacing general coverage immediately.
4. Run `preview_calibration_corpus()` and inspect warnings.
5. Calibrate with exactly the same settings as the baseline.
6. Compare held-out cosine/RMSE **and** image-generation behavior on fixed seeds/prompts.
7. Only promote a custom bridge when both representation metrics and image behavior improve.
