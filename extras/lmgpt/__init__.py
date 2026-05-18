"""LightWeight MAPF-GPT (lmgpt): compression, distillation, evaluation utilities.

This package consolidates code previously scattered across ``extras/*.py``.
Top-level CLI wrappers live in ``extras/scripts/`` and import from here.

Layout::

    lmgpt/
        checkpoints.py      # I/O for .pt; DRY-helpers
        compression/        # trim, dynamic-int8, AWQ, SmoothQuant
        distillation/       # data, losses (logit+hidden), trainer, finetune
        eval/               # POGEMA harness, comparison tables, latency bench
        utils/              # env capture, run-dir helpers

The original public artifacts (``mapf_gpt/model.py``, ``mapf_gpt/inference.py``,
the upstream training pipeline) are *not* modified by this package.
"""

__all__ = ["checkpoints", "compression", "distillation", "eval", "utils"]
__version__ = "0.1.0"
