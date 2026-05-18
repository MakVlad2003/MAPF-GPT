#!/usr/bin/env python3
"""Trim transformer depth of a MAPF-GPT checkpoint.

Example::

    python extras/scripts/trim.py -i weights/model-2M.pt -o weights/model-2M-L3.pt --n_layer 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401

from lmgpt.compression.trim import trim_checkpoint


def main() -> int:
    p = argparse.ArgumentParser(description="Trim MAPF-GPT transformer depth")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--n_layer", "--n-layer", dest="n_layer", type=int, required=True,
        help="Number of transformer blocks to keep (from the start of the stack)",
    )
    args = p.parse_args()
    info = trim_checkpoint(args.input, args.output, args.n_layer)
    print(f"saved {info['output']} ({info['n_params_M']:.3f}M params, "
          f"n_layer {info['old_n_layer']} -> {info['new_n_layer']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
