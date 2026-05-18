"""Losses for knowledge distillation: action-head logits + intermediate hidden states.

``distill_logits_last``         -- baseline (KL on last position + CE).
``distill_logits_multipos``     -- KL/CE over context positions, weighted.
``HiddenProjector``             -- learnable Linear ``R^{n_embd_s} -> R^{n_embd_t}``
                                   so MSE between projected student and teacher
                                   hidden states is well-defined when widths differ.
``hidden_state_loss``           -- MSE between projected student layers and a
                                   chosen subset of teacher layers.
``DistillLoss``                 -- convenience aggregator.

All losses operate on the same 5-action vocabulary slice that the inference
path uses (``logits[..., :5]``); see ``mapf_gpt.model.GPT.act``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapf_gpt.model import GPT

ACTION_VOCAB = 5  # First 5 vocab indices are the actions {stay, up, down, left, right}


@dataclass
class DistillLossWeights:
    alpha_kl: float = 0.7
    alpha_ce: float = 0.3
    alpha_hid: float = 0.0
    temperature: float = 2.0
    multipos_weight_last: float = 1.0
    multipos_weight_other: float = 0.1
    hidden_layer_pairs: Optional[Sequence[Tuple[int, int]]] = None  # (student_layer, teacher_layer)


def distill_logits_last(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    gt_action: torch.Tensor,
    *,
    temperature: float,
    alpha_kl: float,
    alpha_ce: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL + CE on 5-action logits at the last position.

    ``logits_*``: ``[B, ACTION_VOCAB]``, ``gt_action``: ``[B]`` int64 in 0..4.
    """
    device = logits_s.device
    kl_term = torch.zeros((), device=device)
    ce_term = torch.zeros((), device=device)
    total = torch.zeros((), device=device)

    if alpha_kl > 0:
        log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
        p_t = F.softmax(logits_t / temperature, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
        kl_term = (temperature ** 2) * kl
        total = total + alpha_kl * kl_term

    if alpha_ce > 0:
        ce_term = F.cross_entropy(logits_s, gt_action.long())
        total = total + alpha_ce * ce_term

    return total, kl_term.detach(), ce_term.detach()


def distill_logits_multipos(
    logits_s_full: torch.Tensor,
    logits_t_full: torch.Tensor,
    gt_action: torch.Tensor,
    *,
    temperature: float,
    alpha_kl: float,
    alpha_ce: float,
    weight_last: float,
    weight_other: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Position-weighted KL+CE.

    Inputs have shape ``[B, T, ACTION_VOCAB]``; ``gt_action`` is for the last
    position only (we keep the original supervised target). All other-position
    contributions are KL-only.
    """
    B, T, _ = logits_s_full.shape
    device = logits_s_full.device
    weights = torch.full((T,), float(weight_other), device=device)
    weights[-1] = float(weight_last)
    weights = weights / weights.sum()  # normalised so the loss is on a comparable scale

    kl_term = torch.zeros((), device=device)
    ce_term = torch.zeros((), device=device)
    total = torch.zeros((), device=device)

    if alpha_kl > 0:
        log_p_s = F.log_softmax(logits_s_full / temperature, dim=-1)
        p_t = F.softmax(logits_t_full / temperature, dim=-1)
        # KL per position, sum over vocab, mean over batch.
        kl_per_pos = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(dim=-1).mean(dim=0)
        kl = (kl_per_pos * weights).sum()
        kl_term = (temperature ** 2) * kl
        total = total + alpha_kl * kl_term

    if alpha_ce > 0:
        ce_term = F.cross_entropy(logits_s_full[:, -1, :], gt_action.long())
        total = total + alpha_ce * ce_term

    return total, kl_term.detach(), ce_term.detach()


class HiddenProjector(nn.Module):
    """Learnable Linear ``R^{n_embd_s} -> R^{n_embd_t}`` per (student, teacher) layer pair.

    Initialised to scaled identity-ish (Xavier) so the projection is well-behaved
    at the start of training.
    """

    def __init__(self, n_embd_s: int, n_embd_t: int):
        super().__init__()
        self.proj = nn.Linear(n_embd_s, n_embd_t, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        return self.proj(h_s)


def gather_hidden_states(model: GPT, idx: torch.Tensor) -> List[torch.Tensor]:
    """Run the model and return the post-block hidden states (no logits computed).

    Avoids the ``lm_head`` projection to save FLOPs; matches the pre-LayerNorm
    activations going into each block (``h[i]`` is the output of block ``i``).
    """
    device = idx.device
    b, t = idx.size()
    assert t <= model.config.block_size, "context exceeds model block_size"
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    outs: List[torch.Tensor] = []
    for block in model.transformer.h:
        x = block(x)
        outs.append(x)
    return outs


def default_layer_map(n_s: int, n_t: int) -> List[Tuple[int, int]]:
    """Uniform 1:1 layer mapping: student ``i`` <-> teacher ``round((i+1) * n_t / n_s) - 1``."""
    pairs: List[Tuple[int, int]] = []
    for i in range(n_s):
        j = int(round((i + 1) * n_t / n_s)) - 1
        j = max(0, min(n_t - 1, j))
        pairs.append((i, j))
    return pairs


def hidden_state_loss(
    hs_student: List[torch.Tensor],
    hs_teacher: List[torch.Tensor],
    projectors: Dict[int, HiddenProjector],
    pairs: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """MSE between projected student hidden states and teacher counterparts.

    Both sequences are output of :func:`gather_hidden_states`. We average over
    selected ``(i_s, j_t)`` pairs.
    """
    loss = torch.zeros((), device=hs_student[0].device)
    n = 0
    for i_s, j_t in pairs:
        proj = projectors[i_s]
        h_s = hs_student[i_s]
        h_t = hs_teacher[j_t].detach()
        loss = loss + F.mse_loss(proj(h_s), h_t)
        n += 1
    return loss / max(n, 1)


@dataclass
class DistillLoss:
    """Bundle of weights + (lazily-constructed) hidden projectors.

    Use :meth:`build_projectors` once after model construction (we need student
    and teacher widths to size the projection matrices).
    """

    weights: DistillLossWeights = field(default_factory=DistillLossWeights)
    projectors: Dict[int, HiddenProjector] = field(default_factory=dict)
    layer_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def build_projectors(
        self,
        n_embd_s: int,
        n_embd_t: int,
        n_layer_s: int,
        n_layer_t: int,
        device: torch.device,
    ) -> List[nn.Parameter]:
        pairs = (
            list(self.weights.hidden_layer_pairs)
            if self.weights.hidden_layer_pairs is not None
            else default_layer_map(n_layer_s, n_layer_t)
        )
        self.layer_pairs = pairs
        self.projectors = {
            i_s: HiddenProjector(n_embd_s, n_embd_t).to(device) for i_s, _ in pairs
        }
        return [p for proj in self.projectors.values() for p in proj.parameters()]


__all__ = [
    "ACTION_VOCAB",
    "DistillLossWeights",
    "DistillLoss",
    "HiddenProjector",
    "distill_logits_last",
    "distill_logits_multipos",
    "hidden_state_loss",
    "gather_hidden_states",
    "default_layer_map",
]
