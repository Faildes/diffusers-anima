from types import SimpleNamespace

import torch

from diffusers_anima.pipelines.anima.semantic_prompt import AnimaSemanticPromptFrontend, PROMPT_MODE_DIRECT


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        words = str(text).replace(",", " , ").replace(";", " ; ").split()
        ids = list(range(3, 3 + len(words)))
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            ids = ids[: int(max_length)]
        if kwargs.get("return_tensors") == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor([ids or [2]]),
                attention_mask=torch.ones(1, len(ids or [2]), dtype=torch.long),
            )
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


class GenerateModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids=None, **kwargs):
        self.calls += 1
        prompt = ", ".join(f"tag{i}" for i in range(50)) if self.calls == 1 else "tag0, tag1"
        base_len = int(input_ids.shape[-1])
        total_len = base_len + len(self.tokenizer(prompt).input_ids)
        return torch.arange(total_len, dtype=torch.long).view(1, -1)


class DummyPipe:
    def __init__(self):
        tok = TinyTokenizer()
        self.prompt_tokenizer = SimpleNamespace(qwen_tokenizer=tok, t5_tokenizer=tok)
        self.text_encoder = GenerateModel(tok)
        self.processor = None

    def set_prompt_processor(self, processor):
        self.processor = processor

    def clear_prompt_processor(self):
        self.processor = None


def test_semantic_frontend_installs_and_retries_to_budget():
    pipe = DummyPipe()
    frontend = AnimaSemanticPromptFrontend(
        pipe,
        mode=PROMPT_MODE_DIRECT,
        target_t5_tokens=32,
        allow_generation=True,
        compression_retries=1,
    )
    frontend.install()
    assert pipe.processor is frontend
    result = frontend.process_one(", ".join(f"tag{i}" for i in range(100)))
    assert result.used_generation is True
    assert result.anima_qwen_tokens <= 32
    assert result.anima_t5_tokens <= 32
    assert pipe.text_encoder.calls >= 1
