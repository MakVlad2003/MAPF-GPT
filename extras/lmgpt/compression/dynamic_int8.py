"""PyTorch dynamic int8 quantization of ``nn.Linear`` layers.

Important: PyTorch dynamic int8 runs on CPU. On modern GPUs (incl. H200) it is
**slower** than FP32 on the same hardware because the model is moved back to
host. We keep this method as a baseline / negative example - real GPU speedups
require AWQ (W4A16) or SmoothQuant (W8A8), see sibling modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:  # pragma: no cover
    from torch.quantization import quantize_dynamic  # type: ignore[no-redef]

from ..checkpoints import (
    break_wte_lm_head_tie,
    build_gpt,
    load_raw_checkpoint,
    strip_prefix_from_state_dict,
)


def quantize_checkpoint(
    input_path: Path | str,
    output_path: Path | str,
) -> Dict[str, Any]:
    """Produce a CPU-only dynamic int8 checkpoint.

    The saved checkpoint sets ``quantized=True`` and ``quantization='dynamic_int8'``
    so that :mod:`mapf_gpt.inference` knows to instantiate the matching module
    structure before calling ``load_state_dict``.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    ckpt = load_raw_checkpoint(in_p)

    model_args = dict(ckpt["model_args"])
    model, _ = build_gpt(model_args)
    sd = strip_prefix_from_state_dict(ckpt["model"])
    model.load_state_dict(sd, strict=False)
    model.eval()

    break_wte_lm_head_tie(model)
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": model_args,
            "quantized": True,
            "quantization": "dynamic_int8",
            "source_checkpoint": str(in_p.resolve()),
        },
        out_p,
    )
    n_bytes = out_p.stat().st_size
    return {
        "input": str(in_p),
        "output": str(out_p),
        "size_MB": n_bytes / 1e6,
        "device": "cpu",
        "kind": "dynamic_int8",
    }


__all__ = ["quantize_checkpoint"]
