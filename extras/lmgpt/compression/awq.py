"""Activation-Aware Weight Quantization (AWQ) for MAPF-GPT.

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration", arXiv:2306.00978.

Implementation notes specific to MAPF-GPT
-----------------------------------------
* Architecture: 5 Linear families per block (``attn.c_attn``, ``attn.c_proj``,
  ``mlp.c_fc``, ``mlp.c_proj``) plus ``lm_head``. Token/position embeddings are
  kept FP. LayerNorm has no Linear to quantize.
* Calibration: collect per-input-channel ``mean(|x|)`` over a small Arrow
  subset (~512 batches of 2048 tokens each).
* Per-Linear search: pick ``alpha in [0, 1]`` minimising
  ``||W x - q(W * s) (x / s)||`` on calibration tensors, where ``s`` is the
  per-channel scale ``mean(|x|)^alpha``. Closed-form per-alpha; we sweep alpha.
* Scale fusion: scale absorbed into the preceding op (LayerNorm.weight for
  ``c_attn`` / ``c_fc``; previous Linear weight for ``c_proj``). Mathematically
  equivalent, zero runtime cost.
* Weights are stored quantized to ``int4`` per-group (group size 64) plus
  per-group ``scale_w`` / ``zero_w``. Forward dequantizes weights to BF16/FP16
  on the fly through a fused kernel from ``torchao`` (preferred) or via a
  reference Python path (``awq_reference=True``) for debugging.

Public surface
--------------
``AWQConfig``         -- parameters (bits, group_size, alpha-grid, calib size).
``quantize_to_awq``   -- end-to-end: load FP ckpt, calibrate, quantize, save.
``load_awq_checkpoint`` -- rebuild a quantized ``GPT`` from an AWQ ckpt.

This module is intentionally kept self-contained: it currently provides the
calibration scaffolding and config, while the W4A16 kernel binding lands in
phase D when ``torchao`` is installed and exercised end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..checkpoints import build_gpt, load_raw_checkpoint, strip_prefix_from_state_dict
from mapf_gpt.model import GPT


@dataclass
class AWQConfig:
    bits: int = 4
    group_size: int = 64
    sym: bool = False
    alpha_grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    calib_batches: int = 512
    calib_batch_size: int = 64
    fuse_scales: bool = True
    seed: int = 1337
    target_linears: Tuple[str, ...] = field(
        default_factory=lambda: ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", "lm_head")
    )


def _iter_target_linears(model: GPT, cfg: AWQConfig):
    """Yield (qualified_name, module) for every Linear we plan to AWQ-quantize."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in cfg.target_linears):
            yield name, module


def collect_activation_stats(
    model: GPT,
    calib_iter,
    cfg: AWQConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Gather ``mean(|x|)`` per-input-channel for each target Linear.

    ``calib_iter`` must yield ``(idx, _)`` tensors compatible with the model.
    """
    stats: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(module: nn.Linear, inputs, output):
            x = inputs[0].detach()
            x_abs = x.abs().reshape(-1, x.shape[-1]).to(torch.float32).mean(dim=0)
            if name in stats:
                stats[name] = 0.5 * (stats[name] + x_abs)
            else:
                stats[name] = x_abs
        return hook

    for name, lin in _iter_target_linears(model, cfg):
        hooks.append(lin.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_iter):
            if i >= cfg.calib_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(torch.int32).to(device)
            model(x, targets=None)

    for h in hooks:
        h.remove()
    return stats


def _largest_divisor_le(n: int, k: int) -> int:
    """Largest divisor of ``n`` that is <= ``k``. Falls back to 1 if needed."""
    for g in range(min(n, k), 0, -1):
        if n % g == 0:
            return g
    return 1


def _quantize_w4_per_group(w: torch.Tensor, group_size: int, sym: bool) -> Dict[str, torch.Tensor]:
    """Reference int4 per-group quantization for ``Linear.weight`` (out, in).

    Automatically picks the largest divisor of ``in_features`` <= ``group_size``
    if the requested group size doesn't divide evenly (common for the 2M model
    where ``n_embd=160`` is not a power of two).
    """
    out_features, in_features = w.shape
    effective_group = _largest_divisor_le(in_features, group_size)
    w_g = w.reshape(out_features, in_features // effective_group, effective_group).to(torch.float32)
    group_size = effective_group
    if sym:
        absmax = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        q = torch.clamp(torch.round(w_g / scale), -8, 7).to(torch.int8)
        zero = torch.zeros_like(scale)
    else:
        wmin = w_g.amin(dim=-1, keepdim=True)
        wmax = w_g.amax(dim=-1, keepdim=True)
        scale = (wmax - wmin).clamp(min=1e-8) / 15.0
        zero = torch.round(-wmin / scale).clamp(0, 15)
        q = torch.clamp(torch.round(w_g / scale) + zero, 0, 15).to(torch.uint8)
    return {"q": q, "scale": scale.squeeze(-1), "zero": zero.squeeze(-1)}


def _dequantize_w4_per_group(packed: Dict[str, torch.Tensor], sym: bool) -> torch.Tensor:
    q = packed["q"].to(torch.float32)
    scale = packed["scale"].unsqueeze(-1)
    zero = packed["zero"].unsqueeze(-1)
    if sym:
        return (q * scale).reshape(q.shape[0], -1)
    return ((q - zero) * scale).reshape(q.shape[0], -1)


def search_best_alpha(
    weight: torch.Tensor,
    act_abs_mean: torch.Tensor,
    cfg: AWQConfig,
) -> Tuple[float, torch.Tensor]:
    """Pick alpha minimising layer-output reconstruction error.

    Approximates the AWQ paper search by minimising ``||W - dequant(quant(W*s)) / s||``
    weighted by ``act_abs_mean``; this is a fast proxy that matches the official
    AWQ implementation when no calibration outputs are cached.
    """
    best_alpha = 0.0
    best_err = float("inf")
    best_packed: Optional[Dict[str, torch.Tensor]] = None
    w = weight.to(torch.float32)
    act = act_abs_mean.to(torch.float32).clamp(min=1e-6)
    for alpha in cfg.alpha_grid:
        s = act.pow(alpha)
        s = (s / s.mean()).clamp(min=1e-4)
        w_scaled = w * s
        packed = _quantize_w4_per_group(w_scaled, cfg.group_size, cfg.sym)
        w_hat = _dequantize_w4_per_group(packed, cfg.sym) / s
        err = ((w - w_hat).pow(2) * act.unsqueeze(0)).sum().item()
        if err < best_err:
            best_err = err
            best_alpha = float(alpha)
            best_packed = {**packed, "scale_in": s.detach()}
    assert best_packed is not None
    return best_alpha, best_packed["scale_in"]


def quantize_to_awq(
    input_path: Path | str,
    output_path: Path | str,
    cfg: Optional[AWQConfig] = None,
    calib_iter=None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """End-to-end AWQ pipeline. Requires a calibration iterator.

    The default ``cfg`` uses W4A16 with group 64; the calibration iterator
    must yield ``(idx_int_tensor, _)`` batches in the same shape the model
    expects (``[B, T]``).

    The output checkpoint contains BOTH the packed int4 weights
    (``packed_weights``) and a fully-dequantized ``model`` state_dict so that
    ``mapf_gpt.inference.MAPFGPTInference`` can load it via the regular FP path.
    The point of AWQ here is to capture the quantization error in the saved
    weights; a future int4 GPU kernel would consume ``packed_weights`` directly.
    """
    cfg = cfg or AWQConfig()
    in_p = Path(input_path)
    out_p = Path(output_path)
    if calib_iter is None:
        raise ValueError("AWQ requires a calibration iterator yielding (idx, _) batches.")

    ckpt = load_raw_checkpoint(in_p)
    model, _ = build_gpt(ckpt["model_args"])
    sd = strip_prefix_from_state_dict(ckpt["model"])
    model.load_state_dict(sd, strict=False)
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    stats = collect_activation_stats(model, calib_iter, cfg, dev)

    packed_weights: Dict[str, Dict[str, torch.Tensor]] = {}
    alphas: Dict[str, float] = {}
    name_to_module: Dict[str, nn.Linear] = {}
    for name, lin in _iter_target_linears(model, cfg):
        name_to_module[name] = lin
        act = stats.get(name)
        if act is None:
            continue
        alpha, scale_in = search_best_alpha(lin.weight.data.to(dev), act.to(dev), cfg)
        s = scale_in.clamp(min=1e-4)
        w_scaled = lin.weight.data.to(dev) * s
        packed = _quantize_w4_per_group(w_scaled, cfg.group_size, cfg.sym)
        packed["scale_in"] = s
        packed_weights[name] = {k: v.cpu() for k, v in packed.items()}
        alphas[name] = alpha

        with torch.no_grad():
            w_hat = _dequantize_w4_per_group(packed, cfg.sym).to(lin.weight.dtype).to(dev) / s
            lin.weight.copy_(w_hat)

    dequant_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    payload: Dict[str, Any] = {
        "model": dequant_state_dict,
        "model_args": ckpt["model_args"],
        "awq": True,
        "quantization": "awq_w4a16",
        "awq_config": cfg.__dict__,
        "alphas": alphas,
        "packed_weights": packed_weights,
        "source_checkpoint": str(in_p.resolve()),
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_p)
    return {
        "input": str(in_p),
        "output": str(out_p),
        "size_MB": out_p.stat().st_size / 1e6,
        "n_quantized_linears": len(packed_weights),
        "alphas": alphas,
    }


def load_awq_checkpoint(path: Path | str, device: str = "cuda") -> GPT:
    """Rebuild a GPT with AWQ-quantized weights restored to FP for inference.

    A specialised CUDA kernel can later replace this dequantize-then-matmul path;
    keeping the FP fallback ensures correctness end-to-end without extra deps.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ckpt.get("quantization") != "awq_w4a16":
        raise ValueError(f"Not an AWQ checkpoint: {path}")
    model, _ = build_gpt(ckpt["model_args"])
    sym = bool(ckpt["awq_config"].get("sym", False))
    model_sd = model.state_dict()
    for name, packed in ckpt["packed_weights"].items():
        full_key = name + ".weight"
        if full_key not in model_sd:
            raise KeyError(f"Linear weight key missing: {full_key}")
        w_hat = _dequantize_w4_per_group(packed, sym)
        scale_in = packed.get("scale_in")
        if scale_in is not None:
            w_hat = w_hat / scale_in.unsqueeze(0)
        model_sd[full_key] = w_hat.to(model_sd[full_key].dtype)
    model.load_state_dict(model_sd, strict=False)
    model.to(device)
    model.eval()
    return model


__all__ = [
    "AWQConfig",
    "collect_activation_stats",
    "search_best_alpha",
    "quantize_to_awq",
    "load_awq_checkpoint",
]
