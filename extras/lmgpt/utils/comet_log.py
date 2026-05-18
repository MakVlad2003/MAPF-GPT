"""Optional Comet ML logging for distillation / fine-tune runs.

API key resolution (first match wins):

1. ``COMET_API_KEY`` environment variable
2. First non-empty line of ``extras/.key_comet`` under the repo root

Set ``COMET_WORKSPACE`` / ``COMET_PROJECT_NAME`` to override defaults.
Disable per run with ``comet_enabled=False`` or CLI ``--no-comet``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

DEFAULT_PROJECT = "LightWeight-MAPF-GPT"


def resolve_comet_api_key(repo_root: Path) -> Optional[str]:
    env_key = os.environ.get("COMET_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = repo_root / "extras" / ".key_comet"
    if key_file.is_file():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return None


def _numeric_metrics(record: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, val in record.items():
        if key in ("type", "iter"):
            continue
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[str(key)] = float(val)
    return out


class CometLogger:
    """Thin wrapper: no-ops when Comet is disabled or unavailable."""

    def __init__(
        self,
        *,
        enabled: bool,
        repo_root: Path,
        run_dir: Path,
        project_name: str,
        workspace: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        run_type: str = "train",
    ):
        self.enabled = enabled
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.project_name = project_name or os.environ.get("COMET_PROJECT_NAME", DEFAULT_PROJECT)
        self.workspace = workspace or os.environ.get("COMET_WORKSPACE")
        self.experiment_name = experiment_name or run_dir.name
        self.tags = list(tags or [])
        self.run_type = run_type
        self._experiment: Any = None
        self._disabled_reason: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._experiment is not None

    def start(self, parameters: Mapping[str, Any]) -> None:
        if not self.enabled:
            self._disabled_reason = "disabled by config/CLI"
            return

        api_key = resolve_comet_api_key(self.repo_root)
        if not api_key:
            self._disabled_reason = "no API key (COMET_API_KEY or extras/.key_comet)"
            print(f"Comet: skipped — {self._disabled_reason}", file=sys.stderr, flush=True)
            return

        try:
            from comet_ml import Experiment
        except ImportError:
            self._disabled_reason = "comet_ml not installed (pip install comet-ml)"
            print(f"Comet: skipped — {self._disabled_reason}", file=sys.stderr, flush=True)
            return

        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "project_name": self.project_name,
            "auto_metric_logging": False,
            "auto_param_logging": False,
        }
        if self.workspace:
            kwargs["workspace"] = self.workspace

        self._experiment = Experiment(**kwargs)
        self._experiment.set_name(self.experiment_name)
        if self.tags:
            self._experiment.add_tags(list(self.tags))
        self._experiment.log_parameter("run_dir", str(self.run_dir.resolve()))
        self._experiment.log_parameter("run_type", self.run_type)
        self._experiment.log_parameters(dict(parameters))
        print(
            f"Comet: experiment '{self.experiment_name}' "
            f"(project={self.project_name}, workspace={self.workspace or 'default'})",
            flush=True,
        )

    def log_metrics(self, record: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self._experiment is None:
            return
        step_id = step if step is not None else record.get("iter")
        if step_id is None:
            return
        for name, value in _numeric_metrics(record).items():
            self._experiment.log_metric(name, value, step=int(step_id))

    def log_asset(self, path: Path, file_name: Optional[str] = None) -> None:
        if self._experiment is None or not path.is_file():
            return
        self._experiment.log_asset(str(path), file_name=file_name or path.name)

    def end(self) -> None:
        if self._experiment is None:
            return
        for name in ("metrics.jsonl", "summary_latest.json", "run_config.json"):
            self.log_asset(self.run_dir / name)
        try:
            self._experiment.end()
        except Exception as exc:  # noqa: BLE001
            print(f"Comet: end() failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self._experiment = None


__all__ = ["CometLogger", "resolve_comet_api_key", "DEFAULT_PROJECT"]
