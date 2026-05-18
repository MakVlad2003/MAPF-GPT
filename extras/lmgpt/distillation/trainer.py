"""End-to-end distillation training loop (config-driven).

Improvements vs the original ``extras/train_distillation.py``:

1. **Hidden-state distillation** (``alpha_hid > 0``) — adds MSE between projected
   student hidden states and teacher hidden states at chosen layer pairs.
   Without this, gradient flows only through the 5-action logits at the last
   position, which is a weak signal for small students.
2. **Multi-position targets** (``multipos=True``) — KL is computed at every
   context position, weighted to keep the last token dominant (default
   ``weight_last=1.0``, ``weight_other=0.1``). Multiplicatively increases the
   amount of training signal per batch.
3. **Mini-rollout eval** (``mini_rollout_*``) — periodically runs a single
   POGEMA episode on a fixed scenario and reports CSR/ISR alongside val-KL/CE.
   Val-loss alone failed to predict benchmark degradation in our earlier runs.
4. **Pure-CE mode** for fine-tuning trimmed checkpoints (``alpha_kl=0,
   alpha_hid=0``) — equivalent to supervised fine-tune from a warm start.

Outputs follow the canonical layout: ``ckpt_best.pt``, ``ckpt_last.pt``,
``ckpt_iter_<N>.pt``, ``metrics.jsonl``, ``summary_latest.json``,
``run_config.json``, ``environment.txt``, ``README.md``.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..checkpoints import (
    build_gpt,
    load_raw_checkpoint,
    strip_prefix_from_state_dict,
)
from ..utils import append_jsonl, json_safe, make_run_dir, setup_run_logging, write_run_config
from ..utils.comet_log import CometLogger
from .data import DistillArrowIterable
from .losses import (
    ACTION_VOCAB,
    DistillLoss,
    DistillLossWeights,
    distill_logits_last,
    distill_logits_multipos,
    gather_hidden_states,
    hidden_state_loss,
)
from mapf_gpt.model import GPT, GPTConfig


@dataclass
class TrainerConfig:
    teacher: str = ""
    student_config: Dict[str, Any] = field(default_factory=dict)
    train_data: str = "dataset/train"
    valid_data: str = "dataset/validation"
    out_dir: str = "extras/runs/distill/run"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 1337
    max_iters: int = 50_000
    batch_size: int = 2048
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 1000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 20
    num_val_batches: int = 50
    train_estimate_batches: int = 20
    max_train_files: Optional[int] = None
    max_valid_files: Optional[int] = None
    resume: Optional[str] = None

    multipos: bool = False
    multipos_weight_last: float = 1.0
    multipos_weight_other: float = 0.1
    alpha_kl: float = 0.7
    alpha_ce: float = 0.3
    alpha_hid: float = 0.0
    temperature: float = 2.0

    mini_rollout_enabled: bool = False
    mini_rollout_interval: int = 2000
    mini_rollout_map: str = "validation-mazes-seed-000"
    mini_rollout_num_agents: int = 32
    mini_rollout_max_steps: int = 128

    init_from: Optional[str] = None  # warm-start path (e.g. trimmed ckpt)

    comet_enabled: bool = True
    comet_project: str = "LightWeight-MAPF-GPT"
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_tags: List[str] = field(default_factory=list)


def _resolve_paths(cfg: TrainerConfig, repo_root: Path) -> Tuple[Path, Path, Path]:
    """Convert string paths to Path objects rooted at the repo if relative."""
    def as_path(p: str) -> Path:
        return Path(p) if os.path.isabs(p) else (repo_root / p).resolve()
    return as_path(cfg.teacher), as_path(cfg.train_data), as_path(cfg.valid_data)


def _load_teacher(path: Path, device: torch.device) -> GPT:
    ckpt = load_raw_checkpoint(path)
    if ckpt.get("quantized"):
        raise RuntimeError(f"Quantized teacher not supported: {path}")
    model, _ = build_gpt(ckpt["model_args"])
    model.load_state_dict(strip_prefix_from_state_dict(ckpt["model"]), strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_student(cfg: Dict[str, Any], device: torch.device, init_from: Optional[str] = None) -> GPT:
    fields = {k: cfg[k] for k in GPTConfig.__dataclass_fields__ if k in cfg}
    gptconf = GPTConfig(**fields)
    model = GPT(gptconf).to(device)
    if init_from:
        ckpt = load_raw_checkpoint(Path(init_from))
        sd = strip_prefix_from_state_dict(ckpt["model"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-start from {init_from}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    return model


def _action_logits_full(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """Return ``[B, T, ACTION_VOCAB]`` action logits over all positions.

    The model's ``forward(idx, targets=None)`` only returns the last position;
    we replicate the lm_head call on every position for multi-position KD.
    """
    device = idx.device
    b, t = idx.size()
    assert t <= model.config.block_size
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)  # [B, T, vocab]
    return logits[..., :ACTION_VOCAB]


def _action_logits_last(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    logits, _ = model(idx, targets=None)
    return logits.squeeze(1)[..., :ACTION_VOCAB]


def _get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return lr * max(it, 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(max_iters - warmup_iters, 1)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def _validate_gt(y_last: torch.Tensor) -> None:
    bad = (y_last < 0) | (y_last > 4)
    if bad.any():
        raise ValueError(f"GT actions must be in 0..4; got invalid: {y_last[bad][:8]}")


def _try_mini_rollout(
    cfg: TrainerConfig,
    student_state_dict: Dict[str, torch.Tensor],
    student_model_args: Dict[str, Any],
    device: str,
) -> Optional[Dict[str, float]]:
    """Run one POGEMA episode on a fixed scenario and return CSR/ISR/SoC/ep_length.

    Returns ``None`` if pogema_toolbox is unavailable or the rollout failed.
    Keeps imports local so distillation works without POGEMA installed.
    """
    if not cfg.mini_rollout_enabled:
        return None
    try:
        import tempfile
        import yaml
        from pogema_toolbox.create_env import Environment
        from pogema_toolbox.registry import ToolboxRegistry
        from pogema_toolbox.run_episode import run_episode

        from create_env import create_eval_env
        from mapf_gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

        repo_root = Path(__file__).resolve().parents[3]
        for maps_file in (repo_root / "eval_configs").rglob("maps.yaml"):
            with open(maps_file) as f:
                ToolboxRegistry.register_maps(yaml.safe_load(f))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        torch.save({"model": student_state_dict, "model_args": student_model_args}, tmp_path)
        try:
            env = create_eval_env(Environment(
                with_animation=False,
                observation_type="MAPF",
                on_target="nothing",
                map_name=cfg.mini_rollout_map,
                max_episode_steps=cfg.mini_rollout_max_steps,
                num_agents=cfg.mini_rollout_num_agents,
                seed=0,
                obs_radius=5,
                collision_system="soft",
            ))
            algo = MAPFGPTInference(MAPFGPTInferenceConfig(
                path_to_weights=str(tmp_path),
                device=device,
                infer_dtype="float32",
            ))
            algo.reset_states()
            results = dict(run_episode(env, algo))
            return {
                "rollout_csr": float(results.get("CSR", 0.0)),
                "rollout_isr": float(results.get("ISR", 0.0)),
                "rollout_soc": float(results.get("SoC", 0.0)),
                "rollout_ep_length": float(results.get("ep_length", 0.0)),
            }
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"mini-rollout failed: {exc}", file=sys.stderr, flush=True)
        return None


def train(cfg: TrainerConfig, repo_root: Path) -> Path:
    """Run a distillation experiment. Returns the run directory path."""
    if cfg.alpha_kl <= 0 and cfg.alpha_ce <= 0 and cfg.alpha_hid <= 0:
        raise ValueError("At least one of alpha_kl / alpha_ce / alpha_hid must be > 0.")

    teacher_path, train_data_path, valid_data_path = _resolve_paths(cfg, repo_root)
    if not train_data_path.is_dir() or not valid_data_path.is_dir():
        raise FileNotFoundError(
            f"train/valid data dirs must exist: train={train_data_path} valid={valid_data_path}"
        )
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if "cuda" in str(device) and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    run_dir = make_run_dir(cfg.out_dir)
    write_run_config(run_dir, asdict(cfg))
    setup_run_logging(run_dir, repo_root)
    (run_dir / "student_config.json").write_text(json.dumps(cfg.student_config, indent=2), encoding="utf-8")

    run_type = "finetune" if cfg.init_from else "distill"
    comet_tags = list(cfg.comet_tags)
    if run_type not in comet_tags:
        comet_tags.append(run_type)
    comet = CometLogger(
        enabled=cfg.comet_enabled,
        repo_root=repo_root,
        run_dir=run_dir,
        project_name=cfg.comet_project,
        workspace=cfg.comet_workspace,
        experiment_name=cfg.comet_experiment_name or run_dir.name,
        tags=comet_tags,
        run_type=run_type,
    )
    comet_params = {k: v for k, v in asdict(cfg).items() if k != "student_config"}
    comet_params["student_config"] = cfg.student_config
    comet_params["teacher_path"] = str(teacher_path)
    comet.start(comet_params)

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
    use_scaler = cfg.dtype == "float16" and device.type == "cuda"
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        scaler = torch.amp.GradScaler(enabled=False)
        autocast_ctx = nullcontext()

    teacher = _load_teacher(teacher_path, device)
    student = _build_student(cfg.student_config, device, init_from=cfg.init_from)

    distill_loss = DistillLoss(
        DistillLossWeights(
            alpha_kl=cfg.alpha_kl,
            alpha_ce=cfg.alpha_ce,
            alpha_hid=cfg.alpha_hid,
            temperature=cfg.temperature,
            multipos_weight_last=cfg.multipos_weight_last,
            multipos_weight_other=cfg.multipos_weight_other,
        )
    )
    hidden_params: List[torch.nn.Parameter] = []
    if cfg.alpha_hid > 0:
        hidden_params = distill_loss.build_projectors(
            n_embd_s=student.config.n_embd,
            n_embd_t=teacher.config.n_embd,
            n_layer_s=student.config.n_layer,
            n_layer_t=teacher.config.n_layer,
            device=device,
        )
        print(f"hidden distillation enabled with layer pairs: {distill_loss.layer_pairs}", flush=True)

    iter_start = 0
    best_val = float("inf")
    resume_path: Optional[Path] = None
    if cfg.resume:
        rp = Path(cfg.resume)
        if rp.is_dir():
            rp = rp / "ckpt_last.pt"
        resume_ckpt = torch.load(rp, map_location=device, weights_only=False)
        student.load_state_dict(strip_prefix_from_state_dict(resume_ckpt["model"]), strict=False)
        iter_start = int(resume_ckpt.get("iter_num", 0))
        best_val = float(resume_ckpt.get("best_val_loss", float("inf")))
        resume_path = rp
        print(f"resumed from {rp} at iter={iter_start}", flush=True)

    optimizer = student.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device.type
    )
    if hidden_params:
        optimizer.add_param_group({"params": hidden_params, "weight_decay": 0.0})

    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])

    print(f"Student parameters: {student.get_num_params() / 1e6:.4f}M", flush=True)
    print(f"Teacher parameters: {teacher.get_num_params() / 1e6:.4f}M", flush=True)

    train_loader = DistillArrowIterable(
        str(train_data_path), device, cfg.batch_size,
        max_files=cfg.max_train_files, seed_shuffle=cfg.seed,
    )
    val_loader = DistillArrowIterable(
        str(valid_data_path), device, cfg.batch_size,
        max_files=cfg.max_valid_files, seed_shuffle=cfg.seed + 1,
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    metrics_path = run_dir / "metrics.jsonl"
    min_lr = cfg.learning_rate * cfg.min_lr_ratio

    def save_checkpoint(it: int, *, is_best: bool = False, periodic: bool = False) -> None:
        payload = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": it,
            "best_val_loss": best_val,
            "model_args": asdict(student.config),
            "student_config": cfg.student_config,
            "teacher_path": str(teacher_path),
            "distillation": {
                "alpha_kl": cfg.alpha_kl,
                "alpha_ce": cfg.alpha_ce,
                "alpha_hid": cfg.alpha_hid,
                "temperature": cfg.temperature,
                "multipos": cfg.multipos,
            },
        }
        torch.save(payload, run_dir / "ckpt_last.pt")
        if is_best:
            torch.save(payload, run_dir / "ckpt_best.pt")
        if periodic:
            torch.save(payload, run_dir / f"ckpt_iter_{it}.pt")

    def evaluate(it: int) -> Dict[str, float]:
        student.eval()
        totals, kl_terms, ce_terms, acc_gt, acc_t = [], [], [], [], []
        with torch.no_grad():
            for _ in range(cfg.num_val_batches):
                x, y = next(val_iter)
                x = x.to(torch.int32)
                y_last = y[:, -1].to(torch.long)
                _validate_gt(y_last)
                with autocast_ctx:
                    if cfg.multipos:
                        lt = _action_logits_full(teacher, x)
                        ls = _action_logits_full(student, x)
                        total, kl, ce = distill_logits_multipos(
                            ls, lt, y_last,
                            temperature=cfg.temperature,
                            alpha_kl=cfg.alpha_kl, alpha_ce=cfg.alpha_ce,
                            weight_last=cfg.multipos_weight_last,
                            weight_other=cfg.multipos_weight_other,
                        )
                        pred_s = ls[:, -1, :].argmax(dim=-1)
                        pred_t = lt[:, -1, :].argmax(dim=-1)
                    else:
                        lt = _action_logits_last(teacher, x)
                        ls = _action_logits_last(student, x)
                        total, kl, ce = distill_logits_last(
                            ls, lt, y_last,
                            temperature=cfg.temperature,
                            alpha_kl=cfg.alpha_kl, alpha_ce=cfg.alpha_ce,
                        )
                        pred_s = ls.argmax(dim=-1)
                        pred_t = lt.argmax(dim=-1)
                totals.append(float(total.item()))
                kl_terms.append(float(kl.item()) if cfg.alpha_kl > 0 else 0.0)
                ce_terms.append(float(ce.item()) if cfg.alpha_ce > 0 else 0.0)
                acc_gt.append(float((pred_s == y_last).float().mean().item()))
                acc_t.append(float((pred_s == pred_t).float().mean().item()))
        student.train()
        out = {
            "iter": it,
            "val_loss": float(np.mean(totals)),
            "val_kl": float(np.mean(kl_terms)),
            "val_ce": float(np.mean(ce_terms)),
            "val_acc_gt": float(np.mean(acc_gt)),
            "val_teacher_agree": float(np.mean(acc_t)),
        }
        (run_dir / "summary_latest.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        return out

    # main loop
    student.train()
    t0 = time.monotonic()
    x, y = next(train_iter)
    print(
        f"distill loop started (iters {iter_start + 1}..{cfg.max_iters}); "
        f"eval every {cfg.eval_interval}, save every {cfg.save_interval}",
        flush=True,
    )

    try:
        for it in range(iter_start + 1, cfg.max_iters + 1):
            lr = _get_lr(it, cfg.warmup_iters, cfg.max_iters, cfg.learning_rate, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            last_total = torch.tensor(0.0, device=device)

            for _ in range(cfg.grad_accum_steps):
                x = x.to(torch.int32)
                y_last = y[:, -1].to(torch.long)
                _validate_gt(y_last)
                with autocast_ctx:
                    with torch.no_grad():
                        if cfg.multipos:
                            lt = _action_logits_full(teacher, x)
                        else:
                            lt = _action_logits_last(teacher, x)
                        if cfg.alpha_hid > 0:
                            hs_t = gather_hidden_states(teacher, x)
                    if cfg.multipos:
                        ls = _action_logits_full(student, x)
                        total, _, _ = distill_logits_multipos(
                            ls, lt, y_last,
                            temperature=cfg.temperature,
                            alpha_kl=cfg.alpha_kl, alpha_ce=cfg.alpha_ce,
                            weight_last=cfg.multipos_weight_last,
                            weight_other=cfg.multipos_weight_other,
                        )
                    else:
                        ls = _action_logits_last(student, x)
                        total, _, _ = distill_logits_last(
                            ls, lt, y_last,
                            temperature=cfg.temperature,
                            alpha_kl=cfg.alpha_kl, alpha_ce=cfg.alpha_ce,
                        )
                    if cfg.alpha_hid > 0:
                        hs_s = gather_hidden_states(student, x)
                        hid = hidden_state_loss(
                            hs_s, hs_t, distill_loss.projectors, distill_loss.layer_pairs
                        )
                        total = total + cfg.alpha_hid * hid
                    loss_scaled = total / cfg.grad_accum_steps

                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                last_total = total.detach()
                x, y = next(train_iter)

            if cfg.grad_clip > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
                if hidden_params:
                    torch.nn.utils.clip_grad_norm_(hidden_params, cfg.grad_clip)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if cfg.log_interval > 0 and it % cfg.log_interval == 0:
                train_row = {
                    "type": "progress",
                    "iter": it,
                    "train_batch_loss": float(last_total.item()),
                    "lr": lr,
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
                print(json.dumps(train_row), flush=True)
                comet.log_metrics(train_row, step=it)

            if it % cfg.eval_interval == 0:
                va = evaluate(it)
                extra: Dict[str, Any] = {"iter": it, "lr": lr, "time_s": time.monotonic() - t0}
                extra.update(va)
                if cfg.mini_rollout_enabled and it % cfg.mini_rollout_interval == 0:
                    roll = _try_mini_rollout(
                        cfg, student.state_dict(), asdict(student.config), cfg.device
                    )
                    if roll:
                        extra.update(roll)
                append_jsonl(metrics_path, extra)
                print(json.dumps(json_safe(extra)), flush=True)
                comet.log_metrics(extra, step=it)
                if va["val_loss"] < best_val:
                    best_val = va["val_loss"]
                    save_checkpoint(it, is_best=True)
                    comet.log_asset(run_dir / "ckpt_best.pt", file_name="ckpt_best.pt")

            if it % cfg.save_interval == 0:
                save_checkpoint(it, periodic=True)

        save_checkpoint(cfg.max_iters)
    finally:
        comet.end()

    print(f"distillation done; run_dir={run_dir.resolve()}", flush=True)
    return run_dir


__all__ = ["TrainerConfig", "train"]
