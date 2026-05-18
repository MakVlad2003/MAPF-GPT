#!/usr/bin/env python3
"""Pure forward-pass latency benchmark (no POGEMA env).

Example::

    python extras/scripts/bench_latency.py \\
        --weights weights/model-2M.pt --label 2M_FP32 \\
        --batch-sizes 1 32 128 512 2048 \\
        --warmup-iters 50 --measure-iters 200
"""
from __future__ import annotations

import sys

from _bootstrap import REPO_ROOT  # noqa: F401

from lmgpt.eval.latency_bench import main as latency_main


if __name__ == "__main__":
    raise SystemExit(latency_main(sys.argv[1:]))
