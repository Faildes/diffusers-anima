#!/usr/bin/env python3
"""Print Anima v2/v3/native text-encoder metadata and tensor layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safetensors import safe_open
from diffusers_anima.pipelines.anima.text_encoder_bridge import AnimaTextEncoderBridge
from diffusers_anima.pipelines.anima.text_encoder_conditioner import AnimaTextEncoderConditioner
from diffusers_anima.pipelines.anima.text_encoder_bridge import read_text_encoder_profile_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    metadata = read_text_encoder_profile_metadata(args.profile)
    if metadata.get("format") == "anima_native_text_encoder_v1":
        description = {
            "format": metadata.get("format"),
            "artifact_kind": metadata.get("artifact_kind"),
            "source_family": metadata.get("source_family"),
            "anima_ready": metadata.get("anima_ready"),
            "bridge_required": metadata.get("bridge_required"),
            "native_layer_indices_json": metadata.get("native_layer_indices_json"),
            "native_intermediate_size": metadata.get("native_intermediate_size"),
            "training_corpus_lines": metadata.get("training_corpus_lines"),
            "training_steps": metadata.get("training_steps"),
            "training_policy": metadata.get("training_policy"),
        }
    elif metadata.get("format") == "anima_text_encoder_v3":
        obj = AnimaTextEncoderConditioner.from_file(args.profile)
        description = obj.describe()
    else:
        obj = AnimaTextEncoderBridge.from_file(args.profile)
        description = obj.describe()
    print(json.dumps(description, indent=2, ensure_ascii=False))
    with safe_open(str(args.profile), framework="pt", device="cpu") as handle:
        print("\nTensor namespaces:")
        counts = {"bridge": 0, "head": 0, "native": 0, "encoder": 0, "other": 0}
        for key in handle.keys():
            shape = tuple(handle.get_tensor(key).shape)
            namespace = "other"
            for candidate in ("bridge", "head", "native", "encoder"):
                if key.startswith(candidate + "."):
                    namespace = candidate
                    break
            counts[namespace] += 1
            if namespace in {"bridge", "head", "native"}:
                print(f"  {key}: {shape}")
        print(", ".join(f"{k} tensors={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
