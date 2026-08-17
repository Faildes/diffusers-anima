from pathlib import Path


def test_final_encoder_v3_loader_and_finalizer_are_wired():
    loading = Path("src/diffusers_anima/pipelines/anima/loading.py").read_text(encoding="utf-8")
    pipeline = Path("src/diffusers_anima/pipelines/anima/pipeline_anima.py").read_text(encoding="utf-8")
    finalizer = Path("scripts/finalize_anima_text_encoder.py").read_text(encoding="utf-8")
    assert "anima_text_encoder_v3" in loading
    assert "_anima_embedded_conditioner_path" in loading
    assert "load_text_encoder_conditioner" in pipeline
    assert "semantic_expansion" in pipeline
    assert "finalize_anima_text_encoder" in finalizer
