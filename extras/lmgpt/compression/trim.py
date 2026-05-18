"""Depth pruning: keep the first ``n_layer`` transformer blocks.

Used as a structural compression baseline. Quality recovery requires fine-tuning
(see :mod:`lmgpt.distillation.finetune`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ..checkpoints import (
    build_gpt,
    load_raw_checkpoint,
    save_gpt_checkpoint,
    strip_prefix_from_state_dict,
)


def trim_checkpoint(
    input_path: Path | str,
    output_path: Path | str,
    new_n_layer: int,
) -> Dict[str, Any]:
    """Truncate the transformer stack to ``new_n_layer`` blocks.

    Token / position embeddings, LayerNorm and the LM head are preserved bit-exact.
    Returns a small summary dict useful for logging.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    ckpt = load_raw_checkpoint(in_p)

    old_args = dict(ckpt["model_args"])
    old_n = int(old_args["n_layer"])
    if new_n_layer < 1 or new_n_layer > old_n:
        raise ValueError(f"n_layer must be in [1, {old_n}], got {new_n_layer}")

    new_args = {**old_args, "n_layer": new_n_layer}
    model, _ = build_gpt(new_args)

    old_sd = strip_prefix_from_state_dict(ckpt["model"])
    new_sd = model.state_dict()
    missing = []
    for k in new_sd:
        if k not in old_sd:
            missing.append(k)
            continue
        new_sd[k] = old_sd[k].clone()
    if missing:
        raise KeyError(
            f"Old checkpoint missing keys required by trimmed model: {missing[:5]}..."
        )

    model.load_state_dict(new_sd)
    extras: Dict[str, Any] = {
        "trimmed_from": str(in_p.resolve()),
        "original_n_layer": old_n,
    }
    for k in ("iter_num", "best_val_loss"):
        if k in ckpt:
            extras[k] = ckpt[k]
    save_gpt_checkpoint(out_p, model, model_args=new_args, extras=extras)

    return {
        "input": str(in_p),
        "output": str(out_p),
        "old_n_layer": old_n,
        "new_n_layer": new_n_layer,
        "n_params_M": model.get_num_params() / 1e6,
    }


__all__ = ["trim_checkpoint"]
