#!/usr/bin/env python3
"""Aggregate one or more POGEMA evaluation runs into comparison tables.

Thin wrapper for :mod:`lmgpt.eval.compare_runs`. Pass run directories via
``--runs`` (each must contain ``raw_results.jsonl`` from
``extras/scripts/eval_pogema.py``).
"""
from __future__ import annotations

import sys

from _bootstrap import REPO_ROOT  # noqa: F401

from lmgpt.eval.compare_runs import main as compare_main


if __name__ == "__main__":
    raise SystemExit(compare_main(sys.argv[1:]))
