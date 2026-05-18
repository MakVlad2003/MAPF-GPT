#!/usr/bin/env python3
"""SmoothQuant transform (W8A8 candidate) of a MAPF-GPT checkpoint.

Currently emits an FP checkpoint with smoothed weights (no int8 kernel binding
in this phase); useful as a quality ablation against AWQ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from _bootstrap import REPO_ROOT

from lmgpt.compression.smoothquant import SmoothQuantConfig, smoothquant_checkpoint
from lmgpt.distillation.data import DistillArrowIterable


def main() -> int:
    p = argparse.ArgumentParser(description="SmoothQuant transform for MAPF-GPT.")
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--calib-data", type=Path, default=Path("dataset/validation"))
    p.add_argument("--calib-batches", type=int, default=256)
    p.add_argument("--calib-batch-size", type=int, default=32)
    p.add_argument("--calib-max-files", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    calib_path = args.calib_data if args.calib_data.is_absolute() else REPO_ROOT / args.calib_data
    if not calib_path.is_dir():
        print(f"ERROR: calib-data {calib_path} not a directory", file=sys.stderr)
        return 1

    cfg = SmoothQuantConfig(
        alpha=args.alpha,
        calib_batches=args.calib_batches,
        calib_batch_size=args.calib_batch_size,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if "cuda" in str(device) and not torch.cuda.is_available():
        device = torch.device("cpu")

    loader = DistillArrowIterable(
        str(calib_path), device, batch_size=args.calib_batch_size, max_files=args.calib_max_files,
    )
    info = smoothquant_checkpoint(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        calib_iter=iter(loader),
        device=str(device),
    )
    print(f"saved {info['output']} ({info['size_MB']:.2f} MB, "
          f"{info['n_smoothed_linears']} Linears smoothed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
