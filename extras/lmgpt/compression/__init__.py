"""Compression methods for MAPF-GPT checkpoints.

Modules
-------
- :mod:`trim`         — depth pruning (keep first ``n_layer`` blocks).
- :mod:`dynamic_int8` — PyTorch dynamic int8 (CPU-only; kept as baseline / antipattern).
- :mod:`awq`          — Activation-Aware Weight Quantization (W4A16). Phase D.
- :mod:`smoothquant`  — SmoothQuant (W8A8). Phase D, optional.
"""
from __future__ import annotations

from . import trim  # noqa: F401
from . import dynamic_int8  # noqa: F401
from . import awq  # noqa: F401
from . import smoothquant  # noqa: F401

__all__ = ["trim", "dynamic_int8", "awq", "smoothquant"]
