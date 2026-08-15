from __future__ import annotations

from diffusers_anima.pipelines.anima.pipeline_anima import AnimaPipeline


def test_prompt_processor_batch_protocol_without_constructing_pipeline() -> None:
    pipe = object.__new__(AnimaPipeline)

    class Processor:
        def process_batch(self, prompts, *, negative=False):
            prefix = "NEG:" if negative else "POS:"
            return [prefix + item for item in prompts]

    pipe.prompt_processor = Processor()
    assert pipe._process_prompt_batch(["a", "b"], negative=False) == ["POS:a", "POS:b"]
    assert pipe._process_prompt_batch(["x"], negative=True) == ["NEG:x"]


def test_prompt_processor_callable_protocol() -> None:
    pipe = object.__new__(AnimaPipeline)
    pipe.prompt_processor = lambda text, negative=False: ("N" if negative else "P") + text
    assert pipe._process_prompt_batch(["a"], negative=False) == ["Pa"]
    assert pipe._process_prompt_batch(["a"], negative=True) == ["Na"]
