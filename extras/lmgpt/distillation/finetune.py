"""Fine-tuning helpers for trimmed checkpoints (CE and CE+KL self-distillation).

Two flavours, both implemented as thin wrappers on top of :func:`trainer.train`:

* **CE-only** (``alpha_kl=0``, ``alpha_ce=1``): plain supervised fine-tune of a
  trimmed student. Equivalent to one extra epoch of the upstream ``train.py``
  but limited to the post-trim weights and standardised under our run-dir layout.
* **CE+KL self-distillation** (``alpha_kl>0``, ``alpha_ce>0``): use the
  *original* (pre-trim) checkpoint as teacher and the trimmed checkpoint as a
  warm-started student. This is the trim variant that recovered quality in the
  colleague's earlier experiments and we want to make reproducible.

Hidden-state distillation is *off* by default during finetune because student
and teacher share the same width — the gradient comes mainly from logits.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .trainer import TrainerConfig, train


def finetune_trimmed(
    *,
    trimmed_ckpt: str,
    teacher_ckpt: str,
    train_data: str,
    valid_data: str,
    out_dir: str,
    mode: str,  # "ce" | "ce_kl"
    repo_root: Path,
    **overrides: Any,
) -> Path:
    """Run fine-tuning on a trimmed checkpoint.

    ``mode="ce"``     -> alpha_kl=0, alpha_ce=1, teacher unused (we still set a
                          path because TrainerConfig requires one; teacher
                          forward is cheap and stable, so we just ignore its
                          output by zeroing the KL weight).
    ``mode="ce_kl"``  -> alpha_kl=0.7, alpha_ce=0.3 (defaults). ``teacher_ckpt``
                          is typically the *pre-trim* model.
    """
    import torch
    trimmed_path = Path(trimmed_ckpt) if Path(trimmed_ckpt).is_absolute() else repo_root / trimmed_ckpt
    if not trimmed_path.is_file():
        raise FileNotFoundError(f"trimmed checkpoint missing: {trimmed_path}")
    ckpt_args = torch.load(trimmed_path, map_location="cpu", weights_only=False)["model_args"]
    student_cfg: Dict[str, Any] = dict(ckpt_args)

    cfg_kwargs: Dict[str, Any] = dict(
        teacher=teacher_ckpt,
        student_config=student_cfg,
        train_data=train_data,
        valid_data=valid_data,
        out_dir=out_dir,
        init_from=trimmed_ckpt,
        alpha_hid=0.0,
    )
    if mode == "ce":
        cfg_kwargs.update(alpha_kl=0.0, alpha_ce=1.0, temperature=1.0)
    elif mode == "ce_kl":
        cfg_kwargs.update(alpha_kl=0.7, alpha_ce=0.3, temperature=2.0)
    else:
        raise ValueError(f"mode must be 'ce' or 'ce_kl', got {mode!r}")

    cfg_kwargs.update(overrides)
    cfg = TrainerConfig(**cfg_kwargs)
    return train(cfg, repo_root)


__all__ = ["finetune_trimmed"]
