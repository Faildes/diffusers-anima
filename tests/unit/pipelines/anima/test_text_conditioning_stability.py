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
