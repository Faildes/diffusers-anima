from types import SimpleNamespace

import torch

from diffusers_anima.pipelines.anima.text_encoding import (
    AnimaPromptTokenizer,
    resolve_text_encoder_conditioning_scale,
)


class _Tokenizer:
    def __init__(self, *, eos_token_id=None, pad_token_id=0):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(dict(kwargs))
        max_length = int(kwargs.get("max_length", 9999))
        # Simulate an overlength prompt and respect truncation/max_length.
        length = min(700, max_length) if kwargs.get("truncation") else 700
        return SimpleNamespace(input_ids=torch.arange(length).view(1, -1))


def test_native_prompt_tokenizer_truncates_before_encoding():
    qwen = _Tokenizer(pad_token_id=0)
    t5 = _Tokenizer(eos_token_id=1, pad_token_id=0)
    tok = AnimaPromptTokenizer(qwen_tokenizer=qwen, t5_tokenizer=t5)
    out = tok.tokenize_with_weights("x" * 10000)
    assert len(out["qwen"][0]) == 512
    assert len(out["t5xxl"][0]) == 512
    assert qwen.calls[-1]["truncation"] is True
    assert qwen.calls[-1]["max_length"] == 512
    assert t5.calls[-1]["truncation"] is True
    assert t5.calls[-1]["max_length"] == 511


def test_qwen35_default_source_gate_is_conservative():
    encoder = SimpleNamespace(
        _anima_text_encoder_family="qwen3.5",
        config=SimpleNamespace(model_type="qwen3_5_text"),
    )
    assert resolve_text_encoder_conditioning_scale(encoder) == 0.80


def test_qwen3_historical_source_gate_stays_exact():
    encoder = SimpleNamespace(
        _anima_text_encoder_family="qwen3",
        config=SimpleNamespace(model_type="qwen3"),
    )
    assert resolve_text_encoder_conditioning_scale(encoder) == 1.0


class _WordTokenizer:
    def __init__(self, *, eos_token_id=None, pad_token_id=0, multiplier=1):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.multiplier = int(multiplier)
        self.last_text = None

    def __call__(self, text, **kwargs):
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        self.last_text = str(text)
        count = len(str(text).replace(",", " ").split()) * self.multiplier
        ids = list(range(max(1, count)))
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            ids = ids[: int(max_length)]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def test_native_budget_fits_one_shared_text_for_qwen_and_t5():
    # T5 is deliberately denser than Qwen so independent token slicing would
    # correspond to different textual tails.
    qwen = _WordTokenizer(pad_token_id=0, multiplier=1)
    t5 = _WordTokenizer(eos_token_id=1, pad_token_id=0, multiplier=2)
    tok = AnimaPromptTokenizer(qwen_tokenizer=qwen, t5_tokenizer=t5)
    text = ", ".join(f"tag{i}" for i in range(400))
    report = tok.inspect_text_budget(text)
    assert report["was_fitted"] is True
    assert report["fitted_qwen_tokens"] <= 512
    assert report["fitted_t5_content_tokens"] <= 511
    tok.tokenize_with_weights(text)
    # Both real tokenizer calls receive the same fitted textual prefix.
    assert qwen.last_text == t5.last_text == report["fitted_text"]
