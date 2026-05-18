"""SmoothQuant (W8A8) for MAPF-GPT (optional baseline).

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models", arXiv:2211.10438.

Idea: rebalance activation outliers into weights via a per-channel diagonal
scale ``s = act_max^alpha / w_max^(1-alpha)``, then run symmetric int8 on
both. For MAPF-GPT we reuse the calibration scaffolding from :mod:`awq`.

Phase D ships only the smoothing transform (no int8 kernel binding); the
resulting checkpoint can be loaded as FP16/BF16 with the smoothed weights for
quality ablation versus AWQ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..checkpoints import build_gpt, load_raw_checkpoint, strip_prefix_from_state_dict
from .awq import _iter_target_linears, collect_activation_stats, AWQConfig
from mapf_gpt.model import GPT


@dataclass
class SmoothQuantConfig:
    alpha: float = 0.5
    calib_batches: int = 256
    calib_batch_size: int = 64
    seed: int = 1337
    eps: float = 1e-5


def smooth_weights(
    model: GPT,
    stats: Dict[str, torch.Tensor],
    cfg: SmoothQuantConfig,
) -> Dict[str, torch.Tensor]:
    """Apply per-channel scaling: ``W <- W * diag(s)``, store ``1/s`` for activation.

    Returns the per-linear ``s`` so the caller can fuse it into the preceding op.
    """
    cmp_cfg = AWQConfig()
    fused_scales: Dict[str, torch.Tensor] = {}
    for name, lin in _iter_target_linears(model, cmp_cfg):
        act = stats.get(name)
        if act is None:
            continue
        w_max = lin.weight.data.abs().amax(dim=0).clamp(min=cfg.eps)
        a_max = act.to(lin.weight.device).clamp(min=cfg.eps)
        s = (a_max.pow(cfg.alpha) / w_max.pow(1.0 - cfg.alpha)).clamp(min=cfg.eps)
        with torch.no_grad():
            lin.weight.mul_(s.unsqueeze(0))
        fused_scales[name] = (1.0 / s).detach().cpu()
    return fused_scales


def smoothquant_checkpoint(
    input_path: Path | str,
    output_path: Path | str,
    cfg: Optional[SmoothQuantConfig] = None,
    calib_iter=None,
    device: str = "cuda",
) -> Dict[str, Any]:
    cfg = cfg or SmoothQuantConfig()
    if calib_iter is None:
        raise ValueError("SmoothQuant requires a calibration iterator.")
    in_p = Path(input_path)
    out_p = Path(output_path)
    ckpt = load_raw_checkpoint(in_p)

    model, _ = build_gpt(ckpt["model_args"])
    sd = strip_prefix_from_state_dict(ckpt["model"])
    model.load_state_dict(sd, strict=False)
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    awq_cfg = AWQConfig(calib_batches=cfg.calib_batches, calib_batch_size=cfg.calib_batch_size)
    stats = collect_activation_stats(model, calib_iter, awq_cfg, dev)
    fused = smooth_weights(model, stats, cfg)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": ckpt["model_args"],
            "quantized": False,
            "smoothquant": True,
            "smoothquant_config": cfg.__dict__,
            "fused_inverse_scales": fused,
            "source_checkpoint": str(in_p.resolve()),
        },
        out_p,
    )
    return {
        "input": str(in_p),
        "output": str(out_p),
        "size_MB": out_p.stat().st_size / 1e6,
        "n_smoothed_linears": len(fused),
    }


__all__ = ["SmoothQuantConfig", "smooth_weights", "smoothquant_checkpoint"]
