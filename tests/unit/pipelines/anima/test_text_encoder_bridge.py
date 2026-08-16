from types import SimpleNamespace

import torch

from diffusers_anima.pipelines.anima.prompt_plan import AnimaPromptPlan, AnimaPromptSpan
from diffusers_anima.pipelines.anima.text_encoder_bridge import AnimaTextEncoderBridge
from diffusers_anima.pipelines.anima.text_encoding import AnimaPromptTokenizer


def test_bridge_identity_rotation_and_full_center_shift():
    bridge = AnimaTextEncoderBridge(
        rotation=torch.eye(2),
        source_mean=torch.tensor([1.0, 2.0]),
        target_mean=torch.tensor([10.0, 20.0]),
        center_strength=1.0,
    )
    hidden = torch.tensor([[[2.0, 4.0], [1.0, 2.0]]])
    out = bridge.apply(hidden)
    expected = torch.tensor([[[11.0, 22.0], [10.0, 20.0]]])
    torch.testing.assert_close(out, expected)


def test_prompt_plan_validates_spans():
    plan = AnimaPromptPlan(
        text="red hair, blue eyes",
        spans=(AnimaPromptSpan(0, 8, qwen_factor=1.2, t5_factor=1.1, group=0),),
    ).validated()
    assert plan.spans[0].end == 8


class _Tokenizer:
    def __init__(self, *, eos_token_id=None, pad_token_id=0):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, text, **kwargs):
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        # One integer per whitespace-separated source item.
        ids = list(range(len(str(text).split())))
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            ids = ids[: int(kwargs["max_length"])]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def test_qwen_source_is_not_capped_but_t5_queries_are_bounded_and_distributed():
    qwen = _Tokenizer(pad_token_id=0)
    t5 = _Tokenizer(eos_token_id=999, pad_token_id=0)
    tok = AnimaPromptTokenizer(qwen, t5, qwen_source_max_length=None, t5_query_strategy="uniform")
    text = " ".join(f"tok{i}" for i in range(800))
    out = tok.tokenize_with_weights(text)
    qwen_ids = [item[0] for item in out["qwen"][0]]
    t5_ids = [item[0] for item in out["t5xxl"][0]]
    assert len(qwen_ids) == 800
    assert len(t5_ids) == 512
    assert t5_ids[-1] == 999
    # Uniform query coverage includes content from the tail instead of only the first 511 IDs.
    assert max(t5_ids[:-1]) == 799
