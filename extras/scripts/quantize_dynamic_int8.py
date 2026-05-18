#!/usr/bin/env python3
"""Dynamic int8 quantization of MAPF-GPT (CPU inference only).

Kept as a baseline / antipattern: dynamic int8 in PyTorch is CPU-only and
typically *slower* than FP32 on a GPU because activations are converted
on-the-fly. See ``lmgpt.compression.awq`` for the actually-fast variant.

Example::

    python extras/scripts/quantize_dynamic_int8.py -i weights/model-2M.pt -o weights/model-2M-int8.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401

from lmgpt.compression.dynamic_int8 import quantize_checkpoint


def main() -> int:
    p = argparse.ArgumentParser(description="Dynamic int8 quantize MAPF-GPT checkpoint")
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()
    info = quantize_checkpoint(args.input, args.output)
    print(f"saved {info['output']} ({info['size_MB']:.2f} MB on disk, CPU-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
