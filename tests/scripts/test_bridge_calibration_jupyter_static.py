import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def test_calibrator_exposes_jupyter_api():
    source = (SCRIPTS / "calibrate_text_encoder_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert "BridgeCalibrationConfig" in names
    assert "BridgeCalibrationResult" in names
    assert "calibrate_text_encoder_bridge" in names
    assert "preview_calibration_corpus" in names
    assert "make_jupyter_progress_callback" in names


def test_default_corpus_report_is_balanced_and_clean():
    sys.path.insert(0, str(SCRIPTS))
    try:
        from bridge_calibration_corpus import (
            analyze_bridge_calibration_prompts,
            build_default_bridge_calibration_prompts,
        )
        prompts = build_default_bridge_calibration_prompts(4096, 3571)
        report = analyze_bridge_calibration_prompts(prompts)
        assert report["prompt_lines"] == 4096
        assert report["duplicate_lines"] == 0
        assert report["atomic_lines"] > 0
        assert report["medium_lines"] > 0
        assert report["long_lines"] > 0
        assert report["weighted_syntax_lines"] == 0
        assert report["break_syntax_lines"] == 0
        assert report["and_syntax_lines"] == 0
        assert report["warnings"] == []
    finally:
        if str(SCRIPTS) in sys.path:
            sys.path.remove(str(SCRIPTS))


def test_jupyter_notebook_is_valid_json_and_uses_python_api():
    path = ROOT / "notebooks" / "Calibrate_Anima_Text_Encoder_Bridge.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    joined = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "BridgeCalibrationConfig" in joined
    assert "preview_calibration_corpus" in joined
    assert "calibrate_text_encoder_bridge" in joined
