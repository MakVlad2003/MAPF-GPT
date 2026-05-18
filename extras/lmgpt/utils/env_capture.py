"""Reproducibility metadata capture (git, Python, torch, GPU, nvidia-smi).

Replaces three near-duplicate copies that used to live in
``run_author_baselines.py``, ``train_distillation.py`` and the notebook.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _safe_run(cmd: List[str], timeout: int = 25) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    except Exception as exc:
        return f"(failed to run {cmd}: {exc})"


def get_git_info(repo_root: Path) -> Dict[str, str]:
    return {
        "branch": _safe_run(["git", "-C", str(repo_root), "branch", "--show-current"]).strip(),
        "commit": _safe_run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).strip(),
        "short": _safe_run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"]).strip(),
    }


def collect_environment_text(repo_root: Path) -> str:
    """Compose a multi-section text dump suitable for ``environment.txt`` artifacts."""
    lines: List[str] = []
    lines.append(f"run_timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"hostname: {_safe_run(['hostname']).strip()}")
    lines.append(f"cwd: {os.getcwd()}")
    lines.append(f"repo_root: {repo_root}")

    git = get_git_info(repo_root)
    lines.append("\n## git\n")
    lines.append(f"branch: {git['branch']}")
    lines.append(f"commit: {git['commit']}")
    lines.append(f"short: {git['short']}")

    lines.append("\n## python\n")
    lines.append(_safe_run([sys.executable, "--version"]).strip())

    snippet = (
        "import importlib.metadata as im;\n"
        "def ver(*names):\n"
        "    for n in names:\n"
        "        try: return n + '=' + im.version(n)\n"
        "        except Exception: pass\n"
        "    return 'not found'\n"
        "import torch, numpy as np;\n"
        "print('torch', torch.__version__);\n"
        "print('cuda_available', torch.cuda.is_available());\n"
        "print('cuda_version_torch', getattr(torch.version, 'cuda', None));\n"
        "print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);\n"
        "print('numpy', np.__version__);\n"
        "print(ver('pogema'));\n"
        "print(ver('pogema-toolbox', 'pogema_toolbox'));\n"
        "print(ver('pyarrow'));\n"
        "print(ver('torchao'));\n"
    )
    lines.append(_safe_run([sys.executable, "-c", snippet]).strip())

    lines.append("\n## nvidia-smi\n")
    lines.append(_safe_run(["nvidia-smi"], timeout=20))

    return "\n".join(lines) + "\n"
