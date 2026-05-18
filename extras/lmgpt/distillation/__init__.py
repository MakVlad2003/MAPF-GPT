"""Distillation primitives.

Modules
-------
- :mod:`data`     -- Arrow streaming iterator with optional file caps.
- :mod:`losses`   -- KL+CE+hidden-state losses and the ``HiddenProjector``.
- :mod:`trainer`  -- end-to-end training loop (config-driven).
- :mod:`finetune` -- supervised CE / self-distillation CE+KL fine-tuning.
"""
from __future__ import annotations

from . import data, losses, trainer, finetune  # noqa: F401

__all__ = ["data", "losses", "trainer", "finetune"]
