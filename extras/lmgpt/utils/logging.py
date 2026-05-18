"""JSONL/JSON helpers and run-dir bookkeeping."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .env_capture import collect_environment_text


def json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/object types into plain JSON-serialisable values."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append a single JSON object as one line (atomic-ish: flush after write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(dict(record)), ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def make_run_dir(out_dir_base: str | Path, *, tag: Optional[str] = None) -> Path:
    """Create ``<parent>/<timestamp>_<basename>`` ensuring uniqueness.

    Mirrors the behaviour of ``train_distillation.py`` and ``run_author_baselines.py``
    so that downstream aggregators expect the same layout.
    """
    base = Path(out_dir_base)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{base.name}" if base.name else ts
    if tag:
        name = f"{name}_{tag}"
    run_dir = base.parent / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    out = run_dir / "run_config.json"
    out.write_text(json.dumps(json_safe(dict(config)), indent=2), encoding="utf-8")
    return out


def setup_run_logging(run_dir: Path, repo_root: Path) -> Dict[str, Path]:
    """Materialise the canonical reproducibility files in ``run_dir``.

    Always written: ``environment.txt`` (git + python + nvidia-smi), ``errors.log``
    (initialised empty). Returns the produced paths for caller convenience.
    """
    env_path = run_dir / "environment.txt"
    err_path = run_dir / "errors.log"
    env_path.write_text(collect_environment_text(repo_root), encoding="utf-8")
    err_path.write_text("", encoding="utf-8")
    return {"environment": env_path, "errors": err_path}


def print_kv(prefix: str, **kwargs: Any) -> None:
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    print(prefix + " " + " ".join(parts), flush=True, file=sys.stdout)
