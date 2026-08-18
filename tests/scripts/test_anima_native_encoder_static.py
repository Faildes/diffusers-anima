import ast
from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_native_encoder_module_has_bridge_free_artifact_contract():
    source = (ROOT / "src/diffusers_anima/pipelines/anima/anima_native_text_encoder.py").read_text(encoding="utf-8")
    assert 'anima_native_text_encoder_v1' in source
    assert 'bridge_required' in source
    assert 'token-dependent layer mixer' in source
    ast.parse(source)


def test_loader_recognizes_native_encoder_without_embedded_conditioner():
    source = (ROOT / "src/diffusers_anima/pipelines/anima/loading.py").read_text(encoding="utf-8")
    assert 'native_encoder_format = is_anima_native_text_encoder_metadata(metadata)' in source
    assert 'AnimaNativeQwen35Encoder(backbone, native_head)' in source
    assert 'Native v1 needs no bridge/conditioner attachment' in source


def test_training_script_uses_dual_teacher_knowledge_losses():
    source = (ROOT / "scripts/train_anima_native_text_encoder.py").read_text(encoding="utf-8")
    for marker in (
        'anima_compat_weight',
        'source_geometry_weight',
        'token_geometry_weight',
        'knowledge_gain_weight',
        'distribution_weight',
        'bootstrap_token_weight',
    ):
        assert marker in source
    assert 'reference_relax_for_gain' in source
    assert 'format": _NATIVE_ENCODER_FORMAT_V1' in source
    ast.parse(source)


def test_native_training_corpus_builds_minimal_and_binding_pairs():
    path = ROOT / "scripts/native_training_corpus.py"
    spec = importlib.util.spec_from_file_location("native_training_corpus_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    pair = module.make_minimal_pair("1girl, red hair, sitting, left of a chair")
    assert pair is not None
    mutated, category = pair
    assert mutated != "1girl, red hair, sitting, left of a chair"
    assert category in {"appearance", "pose", "spatial", "count", "camera", "framing", "color", "clothing"}
    bindings = module.build_binding_groups(count=8, seed=1)
    assert len(bindings) == 8
    assert all(len(group.texts) == 2 for group in bindings)
