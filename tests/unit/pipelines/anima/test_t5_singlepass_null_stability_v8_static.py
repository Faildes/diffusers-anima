from pathlib import Path


def test_v9_transformer_restores_vanilla_t5_single_pass_contract():
    source = Path(
        "src/diffusers_anima/models/transformers/modeling_anima_transformer.py"
    ).read_text(encoding="utf-8")
    for marker in (
        "t5_single_pass_full_stream = True",
        "target_null_stability_enabled = False",
        "target_null_counts = None",
        "v9 vanilla-contract full T5 stream",
        "max(512, int(adapted.shape[1]))",
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


def test_v9_text_encoding_default_keeps_full_t5_stream():
    source = Path("src/diffusers_anima/pipelines/anima/text_encoding.py").read_text(encoding="utf-8")
    assert "self.t5_query_max_length = None if t5_query_max_length in (None, 0)" in source
    assert "truncation=False" in source
    assert "every T5 token becomes exactly one" in source
    assert "if len(ids) > _CONDITIONING_MAX_LENGTH:" not in source


def test_v9_pipeline_splits_cfg_only_when_real_lengths_differ():
    source = Path("src/diffusers_anima/pipelines/anima/pipeline_anima.py").read_text(encoding="utf-8")
    assert '"target_conditioning_length": "vanilla_single_pass_full_query_min512"' in source
    assert '"target_paging": False' in source
    assert 'pos_cond.shape[1:] != neg_cond.shape[1:]' in source
    assert 'effective_cfg_batch_mode = "split"' in source


def test_v9_training_removes_256_reference_cliff():
    source = Path("scripts/train_anima_native_text_encoder.py").read_text(encoding="utf-8")
    for marker in (
        "reference_max_length: int = 512",
        "long_reference_min_tokens: int = 257",
        "long_reference_fraction: float = 1.00",
        "long_segment_compat_weight: float = 0.25",
        "def _long_segment_compat_loss(",
        "qwen35_fixed_budget_balanced_best_validation_v9_full_reference_t5_vanilla",
    ):
        assert marker in source
