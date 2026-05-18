#!/usr/bin/env python3
"""Fine-tune a trimmed MAPF-GPT checkpoint (CE-only or CE+KL self-distill).

Example (CE-only)::

    python extras/scripts/finetune_trimmed.py \\
        --trimmed weights/model-2M-L4.pt \\
        --teacher weights/model-2M.pt \\
        --mode ce \\
        --out-dir extras/runs/trim_ft/L4_ce

Example (CE + self-distillation KL)::

    python extras/scripts/finetune_trimmed.py \\
        --trimmed weights/model-2M-L3.pt \\
        --teacher weights/model-2M.pt \\
        --mode ce_kl \\
        --out-dir extras/runs/trim_ft/L3_cekl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT

from lmgpt.distillation.finetune import finetune_trimmed


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune a trimmed MAPF-GPT checkpoint.")
    p.add_argument("--trimmed", type=str, required=True, help="Path to the trimmed .pt to warm-start.")
    p.add_argument("--teacher", type=str, required=True, help="Path to teacher .pt (the original pre-trim model).")
    p.add_argument("--mode", choices=["ce", "ce_kl"], default="ce")
    p.add_argument("--train-data", type=str, default="dataset/train")
    p.add_argument("--valid-data", type=str, default="dataset/validation")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--max-iters", type=int, default=30_000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--num-val-batches", type=int, default=50)
    p.add_argument("--max-train-files", type=int, default=None)
    p.add_argument("--max-valid-files", type=int, default=None)
    p.add_argument("--no-comet", action="store_true", help="Disable Comet ML logging.")
    p.add_argument("--comet-project", type=str, default=None, help="Comet project name.")
    p.add_argument("--comet-workspace", type=str, default=None, help="Comet workspace (team/user).")
    p.add_argument("--comet-experiment-name", type=str, default=None, help="Comet experiment display name.")
    p.add_argument(
        "--comet-tag",
        nargs="*",
        default=[],
        metavar="TAG",
        help="Comet tag(s), e.g. --comet-tag final ablation trim_ft",
    )
    args = p.parse_args()

    try:
        run_dir = finetune_trimmed(
            trimmed_ckpt=args.trimmed,
            teacher_ckpt=args.teacher,
            train_data=args.train_data,
            valid_data=args.valid_data,
            out_dir=args.out_dir,
            mode=args.mode,
            repo_root=REPO_ROOT,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            warmup_iters=args.warmup_iters,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            num_val_batches=args.num_val_batches,
            max_train_files=args.max_train_files,
            max_valid_files=args.max_valid_files,
            comet_enabled=not args.no_comet,
            comet_project=args.comet_project or "LightWeight-MAPF-GPT",
            comet_workspace=args.comet_workspace,
            comet_experiment_name=args.comet_experiment_name,
            comet_tags=list(args.comet_tag) + ["finetune", f"mode_{args.mode}"],
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"RUN_DIR={run_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
