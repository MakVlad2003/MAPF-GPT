#!/usr/bin/env python3
"""POGEMA benchmark evaluation for any MAPF-GPT checkpoint.

Thin wrapper around :mod:`lmgpt.eval.pogema_harness` that handles author
baselines (``--models 2M 6M 85M``) and custom checkpoints
(``--custom-weights ... --custom-model-name ...``) uniformly.

Examples::

    # Author baselines on all POGEMA groups
    python extras/scripts/eval_pogema.py --models 2M 6M

    # Custom distilled checkpoint, full benchmark
    python extras/scripts/eval_pogema.py \\
        --custom-weights extras/runs/distill/.../ckpt_best.pt \\
        --custom-model-name student-2M-distilled-85M

    # Quick smoke run (first group, first map, first num_agents)
    python extras/scripts/eval_pogema.py --models 2M --quick
"""
from __future__ import annotations

import sys

from _bootstrap import REPO_ROOT  # noqa: F401

from lmgpt.eval.pogema_harness import main as harness_main


if __name__ == "__main__":
    raise SystemExit(harness_main(sys.argv[1:]))
