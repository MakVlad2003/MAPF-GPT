#!/usr/bin/env python3
"""Activation-Aware Weight Quantization (AWQ) of a MAPF-GPT checkpoint.

Calibrates on validation Arrow files, picks per-Linear alpha minimising
weighted reconstruction error, and saves a W4A16 checkpoint that can be
loaded via :func:`lmgpt.compression.awq.load_awq_checkpoint`.

Example::

    python extras/scripts/quantize_awq.py \\
        -i weights/model-2M.pt -o weights/model-2M-awq.pt \\
        --calib-data dataset/validation --calib-batches 256 --calib-batch-size 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from _bootstrap import REPO_ROOT

from lmgpt.compression.awq import AWQConfig, quantize_to_awq
from lmgpt.distillation.data import DistillArrowIterable


def _calib_iterator(folder: Path, batch_size: int, max_files: int | None, device: torch.device):
    loader = DistillArrowIterable(str(folder), device, batch_size=batch_size, max_files=max_files)
    return loader


def main() -> int:
    p = argparse.ArgumentParser(description="AWQ (W4A16) quantize MAPF-GPT checkpoint.")
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--calib-data", type=Path, default=Path("dataset/validation"))
    p.add_argument("--calib-batches", type=int, default=256)
    p.add_argument("--calib-batch-size", type=int, default=32)
    p.add_argument("--calib-max-files", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--sym", action="store_true", help="Symmetric quantization (default: asymmetric)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    calib_path = args.calib_data if args.calib_data.is_absolute() else REPO_ROOT / args.calib_data
    if not calib_path.is_dir():
        print(f"ERROR: calib-data {calib_path} not a directory", file=sys.stderr)
        return 1

    cfg = AWQConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        calib_batches=args.calib_batches,
        calib_batch_size=args.calib_batch_size,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if "cuda" in str(device) and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")

    iterator = iter(_calib_iterator(calib_path, args.calib_batch_size, args.calib_max_files, device))
    info = quantize_to_awq(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        calib_iter=iterator,
        device=str(device),
    )
    print(f"saved {info['output']} ({info['size_MB']:.2f} MB, "
          f"{info['n_quantized_linears']} Linears quantized)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
