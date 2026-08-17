#!/usr/bin/env python3
"""Print Anima v2 bridge/profile or v3 final-encoder metadata and tensor layout."""
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
    if metadata.get("format") == "anima_text_encoder_v3":
        obj = AnimaTextEncoderConditioner.from_file(args.profile)
        description = obj.describe()
    else:
        obj = AnimaTextEncoderBridge.from_file(args.profile)
        description = obj.describe()
    print(json.dumps(description, indent=2, ensure_ascii=False))
    with safe_open(str(args.profile), framework="pt", device="cpu") as handle:
        print("\nTensor namespaces:")
        counts = {"bridge": 0, "head": 0, "encoder": 0, "other": 0}
        for key in handle.keys():
            shape = tuple(handle.get_tensor(key).shape)
            namespace = "other"
            for candidate in ("bridge", "head", "encoder"):
                if key.startswith(candidate + "."):
                    namespace = candidate
                    break
            counts[namespace] += 1
            if namespace in {"bridge", "head"}:
                print(f"  {key}: {shape}")
        print(", ".join(f"{k} tensors={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
