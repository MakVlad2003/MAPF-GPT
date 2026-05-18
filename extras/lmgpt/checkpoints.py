"""Checkpoint I/O helpers shared by compression and distillation code.

The MAPF-GPT checkpoints produced by upstream ``train.py`` look like::

    {
        "model": OrderedDict[str, Tensor],       # state_dict, possibly with "_orig_mod." prefix
        "model_args": dict,                      # GPTConfig fields
        "iter_num": int,
        "best_val_loss": float,
        ...
    }

Compressed checkpoints add one of these markers:

    "quantized": True, "quantization": "dynamic_int8"    -> dynamic int8 (CPU only)
    "quantized": True, "quantization": "awq_w4a16"       -> AWQ weight-only
    "trimmed_from": "<path>", "original_n_layer": int    -> depth-trim

This module centralises loaders, prefix-stripping and helpers required for
quantization (notably breaking the wte/lm_head weight tying for int8 layers).
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from mapf_gpt.model import GPT, GPTConfig


def strip_prefix_from_state_dict(
    state_dict: Dict[str, Any], prefix: str = "_orig_mod."
) -> Dict[str, Any]:
    """Remove a ``torch.compile`` / DDP prefix from keys while preserving ``_metadata``.

    ``_metadata`` is critical for quantized state dicts: torch quantization
    modules store per-module quant info there.
    """
    keys = list(state_dict.keys())
    if not any(str(k).startswith(prefix) for k in keys):
        return copy.copy(state_dict)
    new_state_dict: OrderedDict = OrderedDict()
    for k in keys:
        v = state_dict[k]
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    src_meta = getattr(state_dict, "_metadata", None)
    if src_meta:
        new_meta: OrderedDict = OrderedDict()
        for mk in list(src_meta.keys()):
            mv = src_meta[mk]
            if mk.startswith(prefix):
                new_meta[mk[len(prefix):]] = mv
            else:
                new_meta[mk] = mv
        new_state_dict._metadata = new_meta  # type: ignore[attr-defined]
    return new_state_dict


def break_wte_lm_head_tie(model: GPT) -> None:
    """Untie ``transformer.wte`` and ``lm_head`` weights (required for int8 quant).

    The GPT model ties these to halve parameter count; dynamic int8 wants them
    as independent ``nn.Parameter`` tensors so the quantization wrapper sees a
    real Linear weight tensor on ``lm_head``.
    """
    w = model.transformer.wte.weight.data.clone()
    model.transformer.wte.weight = nn.Parameter(w.clone())
    model.lm_head.weight = nn.Parameter(w.clone())


def load_raw_checkpoint(path: Path) -> Dict[str, Any]:
    """Read a .pt file to dict on CPU. Trusts user data."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model" not in ckpt or "model_args" not in ckpt:
        raise ValueError(
            f"Checkpoint {path} missing required keys: need 'model' and 'model_args'."
        )
    return ckpt


def build_gpt(model_args: Dict[str, Any]) -> Tuple[GPT, GPTConfig]:
    """Instantiate a fresh GPT from a model_args dict."""
    fields = {k: model_args[k] for k in GPTConfig.__dataclass_fields__ if k in model_args}
    cfg = GPTConfig(**fields)
    model = GPT(cfg)
    return model, cfg


def load_gpt_from_checkpoint(
    path: Path | str,
    device: str = "cpu",
    eval_mode: bool = True,
) -> Tuple[GPT, Dict[str, Any]]:
    """Load a non-quantized GPT checkpoint. Returns (model, ckpt_dict).

    For quantized / AWQ checkpoints use the dedicated loaders in the
    ``compression`` submodules.
    """
    ckpt = load_raw_checkpoint(Path(path))
    if ckpt.get("quantized"):
        raise ValueError(
            f"Checkpoint {path} is quantized ({ckpt.get('quantization')!r}). "
            "Use lmgpt.compression.<method>.load_quantized() instead."
        )
    model, _ = build_gpt(ckpt["model_args"])
    sd = strip_prefix_from_state_dict(ckpt["model"])
    model.load_state_dict(sd, strict=False)
    model.to(device)
    if eval_mode:
        model.eval()
    return model, ckpt


def save_gpt_checkpoint(
    path: Path | str,
    model: GPT,
    *,
    model_args: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a state_dict + model_args to disk in the MAPF-GPT-compatible format."""
    if model_args is None:
        model_args = asdict(model.config)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "model_args": model_args,
    }
    if extras:
        payload.update(extras)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def num_params_million(model: GPT) -> float:
    """Convenience: parameter count in millions (non-embedding semantics from GPT)."""
    return model.get_num_params() / 1e6
