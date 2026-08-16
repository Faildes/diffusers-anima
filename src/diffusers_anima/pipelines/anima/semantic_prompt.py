"""Inference-only semantic prompt frontend for diffusers-anima.

The frontend keeps Anima's learned 512-position conditioning contract intact
while allowing the selected Qwen Base text encoder to read long user prompts,
resolve optional booru/e621 aliases, and compile them into compact prompts that
fit a smaller soft budget before final encoding. No trainable parameters are
introduced. The same Qwen model already loaded by the pipeline is reused for
prompt compression when it exposes ``generate``. If only an encoder backbone is
available, the frontend falls back to deterministic tag/segment budgeting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)

PROMPT_MODE_AUTO = "auto"
PROMPT_MODE_DIRECT = "direct"
PROMPT_MODE_COMPILE = "compile"
PROMPT_MODE_HYBRID = "hybrid"
_SUPPORTED_MODES = {
    PROMPT_MODE_AUTO,
    PROMPT_MODE_DIRECT,
    PROMPT_MODE_COMPILE,
    PROMPT_MODE_HYBRID,
}

_DEFAULT_SYSTEM_PROMPT = """You are an image-generation prompt compressor for Anima.
Rewrite the user's request into one compact hybrid prompt suitable for Anima text conditioning.
Keep only information explicitly present in the input. Do not invent facts, subjects, poses,
or styles. Merge duplicates, synonyms, and repeated phrases. Keep one coherent interpretation,
not multiple alternative compositions. Prefer concise Danbooru/Gelbooru/e621-style tags when
unambiguous, but keep short natural-language relations for spatial, action, counting, and
subject-reference information that tags alone cannot express. Preserve important information in
this priority order: subject count and identity; visible appearance; clothing; action and pose;
camera and composition; critical environment; lighting; style; decorative details. If the input
is too long, remove the lowest-priority decorative details first. Mention the main subject only
once when possible. Return only the final prompt as compact comma/semicolon separated phrases."""

_TAG_SPLIT_RE = re.compile(r"\s*[,;\n]+\s*")
_SENTENCE_HINT_RE = re.compile(r"[.!?。！？]|\b(?:is|are|was|were|with|while|behind|before|after|because|who|that)\b", re.I)


def _normalize_tag_key(value: str) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower())


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


@dataclass(frozen=True)
class TagResolution:
    source: str
    canonical: str
    implications: tuple[str, ...] = ()
    description: str | None = None


@dataclass
class TagLexiconResolver:
    """Small inference-time tag lexicon; this is data, not a model.

    ``aliases`` maps aliases to canonical tags. ``implications`` maps a canonical
    tag to visually meaningful parent tags. ``descriptions`` can provide a short
    natural-language visual gloss for tags that Anima may not know directly.
    """

    aliases: dict[str, str] = field(default_factory=dict)
    implications: dict[str, tuple[str, ...]] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    max_implications_per_tag: int = 3

    def __post_init__(self) -> None:
        self.aliases = {
            _normalize_tag_key(k): _normalize_tag_key(v)
            for k, v in dict(self.aliases).items()
            if str(k).strip() and str(v).strip()
        }
        normalized_implications: dict[str, tuple[str, ...]] = {}
        for key, values in dict(self.implications).items():
            if isinstance(values, str):
                values = [values]
            normalized_implications[_normalize_tag_key(key)] = tuple(
                _dedupe_preserve_order(_normalize_tag_key(v) for v in values)
            )
        self.implications = normalized_implications
        self.descriptions = {
            _normalize_tag_key(k): str(v).strip()
            for k, v in dict(self.descriptions).items()
            if str(k).strip() and str(v).strip()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "TagLexiconResolver":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tag lexicon JSON must contain an object at the top level.")
        return cls(
            aliases=payload.get("aliases", {}),
            implications=payload.get("implications", {}),
            descriptions=payload.get("descriptions", {}),
            max_implications_per_tag=int(payload.get("max_implications_per_tag", 3)),
        )

    def resolve_tag(self, tag: str) -> TagResolution:
        source = str(tag).strip()
        key = _normalize_tag_key(source)
        canonical = self.aliases.get(key, key)
        implications = self.implications.get(canonical, ())[: self.max_implications_per_tag]
        description = self.descriptions.get(canonical)
        return TagResolution(
            source=source,
            canonical=canonical,
            implications=tuple(implications),
            description=description,
        )

    def expand_tag_prompt(self, text: str) -> str:
        parts = [p for p in _TAG_SPLIT_RE.split(str(text or "")) if p.strip()]
        if not parts:
            return str(text or "").strip()
        expanded: list[str] = []
        for part in parts:
            # Multi-word booru tags are often written with spaces by frontends.
            # Resolve them when the lexicon knows the canonicalised key; preserve
            # genuinely free-form natural-language clauses otherwise.
            key = _normalize_tag_key(part)
            known = key in self.aliases or key in self.implications or key in self.descriptions
            if " " in part.strip() and "_" not in part and not known:
                expanded.append(part.strip())
                continue
            resolved = self.resolve_tag(part)
            expanded.append(resolved.canonical)
            expanded.extend(resolved.implications)
            if resolved.description:
                expanded.append(resolved.description)
        return ", ".join(_dedupe_preserve_order(expanded))


@dataclass(frozen=True)
class SemanticPromptResult:
    original: str
    resolved: str
    compiled: str
    mode: str
    qwen_input_tokens: int
    anima_qwen_tokens: int
    anima_t5_tokens: int
    used_generation: bool


class AnimaSemanticPromptFrontend:
    """Prompt compiler / budget manager that can be installed on AnimaPipeline.

    Parameters are intentionally inference-only.  ``target_t5_tokens`` defaults
    to 480 to leave headroom under Anima's learned 512-token conditioning limit.
    Qwen3-0.6B-Base and Qwen3.5-0.8B-Base profiles are both supported and are
    independent of whether the image transformer has 28 or 40 main blocks.

    ``target_t5_tokens`` is the *soft* joint semantic budget applied to both the
    Qwen and T5 tokenizers after compression. The hard conditioning contract
    remains 512 positions, but smaller soft budgets (for example 128-256) often
    yield more stable Anima outputs than simply filling all 512 positions.
    """

    def __init__(
        self,
        pipe: Any,
        *,
        mode: str = PROMPT_MODE_AUTO,
        target_t5_tokens: int = 256,
        qwen_input_max_tokens: int = 8192,
        compiler_max_new_tokens: int = 640,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        tag_resolver: TagLexiconResolver | None = None,
        process_negative_prompt: bool = False,
        allow_generation: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
        compression_retries: int = 1,
    ) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported prompt mode: {mode}")
        if target_t5_tokens < 32 or target_t5_tokens > 512:
            raise ValueError("target_t5_tokens must be in [32, 512].")
        if qwen_input_max_tokens < 64:
            raise ValueError("qwen_input_max_tokens must be >= 64.")
        self.pipe = pipe
        self.mode = mode
        self.target_t5_tokens = int(target_t5_tokens)
        self.qwen_input_max_tokens = int(qwen_input_max_tokens)
        self.compiler_max_new_tokens = int(compiler_max_new_tokens)
        self.system_prompt = str(system_prompt)
        self.tag_resolver = tag_resolver
        self.process_negative_prompt = bool(process_negative_prompt)
        # Base Qwen checkpoints are used as Anima text encoders.  They are not
        # instruction-tuned prompt compilers, and autoregressive generation can
        # take a long time before denoising starts (especially when Qwen3.5
        # linear-attention kernels fall back to the torch implementation).
        # Keep generation opt-in; deterministic resolver/budget processing stays on.
        self.allow_generation = bool(allow_generation)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.compression_retries = max(0, int(compression_retries))
        self.last_results: list[SemanticPromptResult] = []

    @property
    def qwen_tokenizer(self) -> Any:
        prompt_tokenizer = getattr(self.pipe, "prompt_tokenizer", None)
        tokenizer = getattr(prompt_tokenizer, "qwen_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Anima semantic frontend requires a Qwen tokenizer.")
        return tokenizer

    @property
    def t5_tokenizer(self) -> Any:
        prompt_tokenizer = getattr(self.pipe, "prompt_tokenizer", None)
        tokenizer = getattr(prompt_tokenizer, "t5_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Anima semantic frontend requires the Anima T5 tokenizer resource.")
        return tokenizer

    def install(self) -> "AnimaSemanticPromptFrontend":
        setter = getattr(self.pipe, "set_prompt_processor", None)
        if not callable(setter):
            raise AttributeError(
                "Pipeline does not expose set_prompt_processor(); apply the matching diffusers-anima patch first."
            )
        setter(self)
        return self

    def uninstall(self) -> None:
        clearer = getattr(self.pipe, "clear_prompt_processor", None)
        if callable(clearer):
            clearer()

    def _token_count(self, tokenizer: Any, text: str) -> int:
        # Tokenizers such as Anima's T5 resource advertise model_max_length=512.
        # We intentionally count before applying the semantic budget, so silence
        # the misleading HF warning here; this path does not run the model.
        try:
            encoded = tokenizer(
                str(text),
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
        except TypeError:
            encoded = tokenizer(
                str(text),
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )
        ids = getattr(encoded, "input_ids", None)
        if ids is None and isinstance(encoded, dict):
            ids = encoded.get("input_ids", [])
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids or [])

    def _detect_mode(self, text: str) -> str:
        if self.mode != PROMPT_MODE_AUTO:
            return self.mode
        raw = str(text or "").strip()
        if not raw:
            return PROMPT_MODE_DIRECT
        comma_parts = [p for p in _TAG_SPLIT_RE.split(raw) if p.strip()]
        sentence_like = bool(_SENTENCE_HINT_RE.search(raw))
        avg_words = (
            sum(max(1, len(part.split())) for part in comma_parts) / len(comma_parts)
            if comma_parts else 999.0
        )
        # Booru prompts commonly use multi-word tags ("blue eyes", "looking at viewer"),
        # so comma density and short phrase length are more reliable than underscores.
        tag_like = len(comma_parts) >= 3 and avg_words <= 4.0
        if tag_like and sentence_like:
            return PROMPT_MODE_HYBRID
        if tag_like:
            return PROMPT_MODE_DIRECT
        return PROMPT_MODE_COMPILE

    def _resolve_text(self, text: str) -> str:
        if self.tag_resolver is None:
            return str(text or "").strip()
        return self.tag_resolver.expand_tag_prompt(text)

    def _model_device(self, model: Any) -> torch.device:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_compiler_text(
        self,
        text: str,
        mode: str,
        *,
        target_budget: int,
        retry_index: int = 0,
        previous_attempt: str | None = None,
    ) -> str:
        instruction = self.system_prompt
        instruction += (
            f"\nTarget a compact result that fits within about {int(target_budget)} tokens under Anima's prompt budget."
        )
        if mode == PROMPT_MODE_HYBRID:
            instruction += "\nThe input already contains useful tags: preserve them unless they are duplicates or aliases."
        if retry_index > 0:
            instruction += (
                "\nPrevious output was still too long. Compress further while preserving the highest-priority visual facts."
            )
        tokenizer = self.qwen_tokenizer
        user_text = str(text)
        if previous_attempt:
            user_text += (
                "\n\nPREVIOUS COMPACT PROMPT (still too long):\n"
                f"{previous_attempt}\n\nPlease rewrite it even shorter."
            )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ]
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                try:
                    return str(
                        apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                except TypeError:
                    return str(
                        apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
            except Exception as exc:
                logger.debug("Qwen chat template unavailable; using plain compiler prompt: %s", exc)
        return f"{instruction}\n\nUSER PROMPT:\n{text}\n\nCOMPILED PROMPT:\n"

    def _clean_compiler_output(self, text: str) -> str:
        value = str(text or "").strip()
        value = re.sub(r"<think>.*?</think>", "", value, flags=re.S | re.I).strip()
        value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value).strip()
        value = re.sub(r"^(?:compiled prompt|prompt)\s*:\s*", "", value, flags=re.I)
        return value.strip()

    def _generate_compile(
        self,
        text: str,
        mode: str,
        *,
        target_budget: int,
        retry_index: int = 0,
        previous_attempt: str | None = None,
    ) -> tuple[str, bool, int]:
        tokenizer = self.qwen_tokenizer
        if not self.allow_generation:
            return text, False, self._token_count(tokenizer, text)
        model = getattr(self.pipe, "text_encoder", None)
        if model is None or not callable(getattr(model, "generate", None)):
            return text, False, self._token_count(tokenizer, text)

        compiler_text = self._build_compiler_text(
            text,
            mode,
            target_budget=target_budget,
            retry_index=retry_index,
            previous_attempt=previous_attempt,
        )
        tokenized = tokenizer(
            compiler_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.qwen_input_max_tokens,
            add_special_tokens=True,
        )
        input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
        attention_mask = (
            tokenized.get("attention_mask") if isinstance(tokenized, dict) else getattr(tokenized, "attention_mask", None)
        )
        input_len = int(input_ids.shape[-1])
        device = self._model_device(model)
        input_ids = input_ids.to(device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.compiler_max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        generate_kwargs.update(self.generation_kwargs)
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask.to(device)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if eos_token_id is not None:
            generate_kwargs.setdefault("eos_token_id", int(eos_token_id))
        if pad_token_id is not None:
            generate_kwargs.setdefault("pad_token_id", int(pad_token_id))

        with torch.inference_mode():
            generated = model.generate(input_ids=input_ids, **generate_kwargs)
        continuation = generated[0, input_len:]
        decoded = self._clean_compiler_output(
            tokenizer.decode(continuation, skip_special_tokens=True)
        )
        return (decoded or text), bool(decoded), input_len

    def _effective_budget(self, budget: int | None = None) -> int:
        if budget is None:
            budget = self.target_t5_tokens
        return max(32, min(512, int(budget)))

    def _fits_budget(self, text: str, *, budget: int | None = None) -> bool:
        effective_budget = self._effective_budget(budget)
        return (
            self._token_count(self.qwen_tokenizer, text) <= effective_budget
            and self._token_count(self.t5_tokenizer, text) <= effective_budget
        )

    def _fits_final_budget(self, text: str) -> bool:
        return self._fits_budget(text, budget=self.target_t5_tokens)

    def _fit_budget(self, text: str, *, budget: int | None = None) -> str:
        text = str(text or "").strip()
        if self._fits_budget(text, budget=budget):
            return text

        # Keep complete clauses in prompt-priority order. Do not skip an
        # overflowing middle clause and then append later tags; doing so changes
        # the semantic mixture seen by Qwen/T5 and can create competing layouts.
        segments = _dedupe_preserve_order(_TAG_SPLIT_RE.split(text))
        kept: list[str] = []
        for segment in segments:
            candidate = ", ".join([*kept, segment]).strip()
            if self._fits_budget(candidate, budget=budget):
                kept.append(segment)
                continue
            break
        compact = ", ".join(kept).strip()
        if compact:
            return compact

        # Defensive single-clause fallback: choose one shared textual prefix by
        # binary search so Qwen and T5 never receive independently truncated tails.
        words = text.split()
        pieces = words if len(words) > 1 else list(text)
        lo, hi = 0, len(pieces)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = (
                " ".join(pieces[:mid]).strip()
                if len(words) > 1
                else "".join(pieces[:mid]).strip()
            )
            if not candidate:
                lo = mid + 1
                continue
            if self._fits_budget(candidate, budget=budget):
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best or text[:1]

    def _fit_final_budget(self, text: str) -> str:
        return self._fit_budget(text, budget=self.target_t5_tokens)

    def _fit_hard_budget(self, text: str) -> str:
        return self._fit_budget(text, budget=512)

    def _retry_target_budget(self, retry_index: int) -> int:
        base = self._effective_budget(self.target_t5_tokens)
        return max(32, base - 32 * max(0, int(retry_index)))

    # Backward-compatible private alias used by older integrations/tests.
    def _fit_t5_budget(self, text: str) -> str:
        return self._fit_final_budget(text)

    def process_one(self, text: str, *, negative: bool = False) -> SemanticPromptResult:
        original = str(text or "")
        if negative and not self.process_negative_prompt:
            final = self._fit_final_budget(original)
            return SemanticPromptResult(
                original=original,
                resolved=original,
                compiled=final,
                mode=PROMPT_MODE_DIRECT,
                qwen_input_tokens=self._token_count(self.qwen_tokenizer, original),
                anima_qwen_tokens=self._token_count(self.qwen_tokenizer, final),
                anima_t5_tokens=self._token_count(self.t5_tokenizer, final),
                used_generation=False,
            )

        mode = self._detect_mode(original)
        resolved = self._resolve_text(original)
        qwen_input_tokens = self._token_count(self.qwen_tokenizer, resolved)
        used_generation = False
        compiled = resolved
        needs_compile = mode in {PROMPT_MODE_COMPILE, PROMPT_MODE_HYBRID} or (
            mode == PROMPT_MODE_DIRECT and not self._fits_final_budget(resolved)
        )
        if needs_compile:
            target_budget = self._retry_target_budget(0)
            compiled, used_generation, qwen_input_tokens = self._generate_compile(
                resolved,
                mode,
                target_budget=target_budget,
                retry_index=0,
            )
            previous = compiled
            for retry_index in range(1, self.compression_retries + 1):
                if self._fits_final_budget(previous):
                    break
                retry_budget = self._retry_target_budget(retry_index)
                previous, generated, _ = self._generate_compile(
                    previous,
                    PROMPT_MODE_COMPILE,
                    target_budget=retry_budget,
                    retry_index=retry_index,
                    previous_attempt=previous,
                )
                used_generation = used_generation or generated
            compiled = previous

        compiled = self._fit_final_budget(compiled)
        compiled = self._fit_hard_budget(compiled)
        return SemanticPromptResult(
            original=original,
            resolved=resolved,
            compiled=compiled,
            mode=mode,
            qwen_input_tokens=qwen_input_tokens,
            anima_qwen_tokens=self._token_count(self.qwen_tokenizer, compiled),
            anima_t5_tokens=self._token_count(self.t5_tokenizer, compiled),
            used_generation=used_generation,
        )

    def process_batch(self, prompts: Sequence[str], *, negative: bool = False) -> list[str]:
        results = [self.process_one(text, negative=negative) for text in prompts]
        self.last_results = results
        return [result.compiled for result in results]

    def __call__(self, text: str, *, negative: bool = False) -> str:
        result = self.process_one(text, negative=negative)
        self.last_results = [result]
        return result.compiled


__all__ = [
    "AnimaSemanticPromptFrontend",
    "SemanticPromptResult",
    "TagLexiconResolver",
    "TagResolution",
    "PROMPT_MODE_AUTO",
    "PROMPT_MODE_DIRECT",
    "PROMPT_MODE_COMPILE",
    "PROMPT_MODE_HYBRID",
]
