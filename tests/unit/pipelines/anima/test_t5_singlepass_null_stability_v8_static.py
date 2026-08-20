from pathlib import Path


def test_v8_transformer_is_single_pass_full_t5_with_null_stability():
    source = Path(
        "src/diffusers_anima/models/transformers/modeling_anima_transformer.py"
    ).read_text(encoding="utf-8")
    for marker in (
        "t5_single_pass_full_stream = True",
        "target_null_stability_enabled = True",
        "def target_null_key_counts(",
        "def stable_condition_length(",
        "null_key_counts=target_null_key_counts",
        "one DiT call for the complete ordered conditioning stream",
    ):
        assert marker in source
    # v7 target paging / multi-DiT fusion must not remain active.
    for forbidden in (
        "core_condition_page_size",
        "target_long_context_mode",
        "target_long_context_window_size",
        "target_long_context_overlap",
        "target_density_power",
    ):
        assert forbidden not in source


def test_v8_text_encoding_default_keeps_full_t5_stream():
    source = Path("src/diffusers_anima/pipelines/anima/text_encoding.py").read_text(encoding="utf-8")
    assert "self.t5_query_max_length = None if t5_query_max_length in (None, 0)" in source
    assert "truncation=False" in source
    assert "every T5 token becomes exactly one" in source
    # The old structured-token helper used to hard-cut to 512 regardless of
    # the public unlimited-query setting. That unconditional cut must be gone.
    assert "ids = ids[:512]" not in source


def test_v8_pipeline_splits_cfg_when_stable_lengths_differ():
    source = Path("src/diffusers_anima/pipelines/anima/pipeline_anima.py").read_text(encoding="utf-8")
    assert '"target_conditioning_length": "single_pass_full_query_null_stabilized"' in source
    assert '"target_paging": False' in source
    assert 'pos_cond.shape[1:] != neg_cond.shape[1:]' in source
    assert 'effective_cfg_batch_mode = "split"' in source
