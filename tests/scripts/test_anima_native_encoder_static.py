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
        'binding_geometry_weight',
        'reference_batch_fraction',
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
    categories = {group.category for group in module.build_binding_groups(count=25, seed=1)}
    assert {"binding", "attribute_swap", "count_binding", "multilingual_binding"} <= categories


def test_native_encoder_skips_legacy_missing_bridge_warning():
    source = (ROOT / "src/diffusers_anima/pipelines/anima/pipeline_anima.py").read_text(encoding="utf-8")
    assert 'and not bool(getattr(pipe.text_encoder, "_anima_native_encoder", False))' in source
    ast.parse(source)


def test_cuda_indexed_device_uses_cuda_runtime_paths():
    source = (ROOT / "src/diffusers_anima/pipelines/anima/pipeline_anima.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_execution_device_type", "_resolve_sample_dtype", "_resolve_effective_cfg_batch_mode"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in functions} == wanted

    namespace = {
        "torch": __import__("torch"),
        "DTYPE_MAP": {
            "auto": None,
            "float16": __import__("torch").float16,
            "bfloat16": __import__("torch").bfloat16,
            "float32": __import__("torch").float32,
        },
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<pipeline-runtime-test>", "exec"), namespace)

    torch = namespace["torch"]
    assert namespace["_execution_device_type"]("cuda:0") == "cuda"
    assert namespace["_resolve_sample_dtype"](
        "auto", model_dtype=torch.bfloat16, execution_device="cuda:0"
    ) == torch.bfloat16
    assert namespace["_resolve_effective_cfg_batch_mode"](
        "auto", execution_device="cuda:0"
    ) == "concat"


def test_v3_training_is_fixed_budget_balanced_and_best_validation_driven():
    source = (ROOT / "scripts/train_anima_native_text_encoder.py").read_text(encoding="utf-8")
    for marker in (
        'fixed_budget_steps',
        '_iter_balanced_group_batches',
        'split_validation_lines',
        '_build_validation_targets',
        '_evaluate_validation',
        '_snapshot_trainable_state',
        '_restore_trainable_state',
        'qwen35_fixed_budget_balanced_best_validation_v3',
        'training_best_step',
        'validation_neutral_channel',
    ):
        assert marker in source
    ast.parse(source)


def test_v3_corpus_validation_size_is_not_a_fraction_of_corpus_and_color_controls_exist():
    path = ROOT / "scripts/native_training_corpus.py"
    spec = importlib.util.spec_from_file_location("native_training_corpus_v3_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    lines = [f"1girl, sample {i}, red hair" for i in range(5000)]
    train, val = module.split_validation_lines(lines, validation_size=128, seed=3)
    assert len(val) == 128
    assert len(train) == len(lines) - 128
    assert not ({x.casefold() for x in train} & {x.casefold() for x in val})

    weights = module.default_sampling_bucket_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["binding"] >= weights["color"]
    controls = module.build_color_control_groups(count=25, seed=4)
    assert len(controls) == 25
    assert all(group.category == "color_control" for group in controls)
    joined = "\n".join(text for group in controls for text in group.texts)
    assert "high saturation" in joined
    assert "controlled saturation" in joined
