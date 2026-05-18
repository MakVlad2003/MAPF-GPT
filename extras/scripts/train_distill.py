#!/usr/bin/env python3
"""Knowledge distillation training for MAPF-GPT.

Reads a JSON config (``extras/configs/distill/*.json``) plus CLI overrides
and dispatches to :func:`lmgpt.distillation.trainer.train`.

Example::

    python extras/scripts/train_distill.py \\
        --config extras/configs/distill/with_hidden.json \\
        --student extras/configs/students/student-2M.json \\
        --teacher weights/model-85M.pt \\
        --out-dir extras/runs/distill/2M_from_85M_hidden
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT

from lmgpt.distillation.trainer import TrainerConfig, train


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="Train a distilled MAPF-GPT student.")
    p.add_argument("--config", type=Path, required=True, help="Distillation JSON config (TrainerConfig fields).")
    p.add_argument("--student", type=Path, default=None, help="Override student GPTConfig JSON path.")
    p.add_argument("--teacher", type=str, default=None, help="Override teacher checkpoint path.")
    p.add_argument("--train-data", type=str, default=None)
    p.add_argument("--valid-data", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    p.add_argument("--mini-rollout", action="store_true", help="Enable periodic POGEMA mini-rollout.")
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
        help="Comet tag(s), e.g. --comet-tag final ablation distill",
    )
    args = p.parse_args()

    cfg_dict = _load_json(args.config)
    if args.student:
        cfg_dict["student_config"] = _load_json(args.student)
    elif "student_config_path" in cfg_dict:
        s_path = Path(cfg_dict["student_config_path"])
        if not s_path.is_absolute():
            s_path = REPO_ROOT / s_path
        cfg_dict["student_config"] = _load_json(s_path)

    for cli_key, val in [
        ("teacher", args.teacher),
        ("train_data", args.train_data),
        ("valid_data", args.valid_data),
        ("out_dir", args.out_dir),
        ("max_iters", args.max_iters),
        ("batch_size", args.batch_size),
        ("dtype", args.dtype),
        ("device", args.device),
        ("resume", args.resume),
        ("init_from", args.init_from),
        ("max_train_files", args.max_train_files),
        ("max_valid_files", args.max_valid_files),
    ]:
        if val is not None:
            cfg_dict[cli_key] = val

    if args.mini_rollout:
        cfg_dict["mini_rollout_enabled"] = True

    if args.no_comet:
        cfg_dict["comet_enabled"] = False
    if args.comet_project is not None:
        cfg_dict["comet_project"] = args.comet_project
    if args.comet_workspace is not None:
        cfg_dict["comet_workspace"] = args.comet_workspace
    if args.comet_experiment_name is not None:
        cfg_dict["comet_experiment_name"] = args.comet_experiment_name
    if args.comet_tag:
        cfg_dict["comet_tags"] = list(args.comet_tag)

    valid_keys = {f.name for f in dataclasses.fields(TrainerConfig)}
    unknown = set(cfg_dict) - valid_keys
    if unknown:
        for k in sorted(unknown):
            cfg_dict.pop(k)

    if "student_config" not in cfg_dict or not cfg_dict["student_config"]:
        print("ERROR: student_config must be supplied (via --student or JSON).", file=sys.stderr)
        return 2

    cfg = TrainerConfig(**cfg_dict)

    if args.dry_run:
        print("=== distill dry-run ===")
        print(json.dumps({k: getattr(cfg, k) for k in valid_keys if k != "student_config"}, indent=2, default=str))
        print(f"student_config: {cfg.student_config}")
        return 0

    run_dir = train(cfg, REPO_ROOT)
    print(f"RUN_DIR={run_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
