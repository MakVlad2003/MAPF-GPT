"""Shared sys.path bootstrap so CLI scripts can ``from lmgpt import ...``.

We add the repo root (so ``mapf_gpt`` resolves) and the ``extras/`` directory
(so ``lmgpt`` resolves). Idempotent and safe to import multiple times.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_paths() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    extras_root = here.parents[1]
    for p in (str(repo_root), str(extras_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root


REPO_ROOT = add_paths()
