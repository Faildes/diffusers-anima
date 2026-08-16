from pathlib import Path


def test_v2_profile_format_and_bundled_encoder_loader_exist():
    bridge = Path("src/diffusers_anima/pipelines/anima/text_encoder_bridge.py").read_text(encoding="utf-8")
    loading = Path("src/diffusers_anima/pipelines/anima/loading.py").read_text(encoding="utf-8")
    assert "anima_text_encoder_profile_v2" in bridge
    assert 'key.startswith("encoder.")' in loading
    assert "_anima_embedded_bridge_path" in loading


def test_calibrator_has_builtin_corpus_and_bundle_mode():
    source = Path("scripts/calibrate_text_encoder_bridge.py").read_text(encoding="utf-8")
    assert '"--prompts"' in source
    assert '"--bundle-source-weights"' in source
    assert "build_default_bridge_calibration_prompts" in source
