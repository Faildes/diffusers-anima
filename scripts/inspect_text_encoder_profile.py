#!/usr/bin/env python3
"""Print Anima text-encoder profile metadata and tensor layout."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    bridge = AnimaTextEncoderBridge.from_file(args.profile)
    print(json.dumps(bridge.describe(), indent=2, ensure_ascii=False))
    with safe_open(str(args.profile), framework="pt", device="cpu") as handle:
        print("\nTensor namespaces:")
        bridge_count = encoder_count = other_count = 0
        for key in handle.keys():
            shape = tuple(handle.get_tensor(key).shape)
            if key.startswith("bridge."):
                bridge_count += 1
            elif key.startswith("encoder."):
                encoder_count += 1
            else:
                other_count += 1
            if key.startswith("bridge."):
                print(f"  {key}: {shape}")
        print(f"bridge tensors={bridge_count}, encoder tensors={encoder_count}, other tensors={other_count}")


if __name__ == "__main__":
    main()
