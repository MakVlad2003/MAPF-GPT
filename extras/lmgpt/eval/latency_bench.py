"""Pure forward-pass latency benchmark (no POGEMA environment).

Measures ``model.act(idx)`` time in milliseconds under CUDA events with proper
warmup. Sweeps batch size to approximate cost per agent at increasing
parallelism (one observation tensor per agent at a single environment step).

Output: a CSV row per (model, dtype/quant, batch_size) with p50, p95, mean,
plus tokens/sec and per-batch ms. Writing the rows incrementally so a long
sweep yields partial results even on Ctrl+C.

Distinct from POGEMA ``runtime`` (which is env+policy end-to-end episode time);
use both for a complete picture.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXTRAS_ROOT = SCRIPT_PATH.parents[2]
if str(EXTRAS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRAS_ROOT))

from mapf_gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from lmgpt.utils import make_run_dir, setup_run_logging, write_run_config


@dataclass
class LatencyBenchConfig:
    weights: str = ""
    label: str = ""
    device: str = "cuda"
    dtype: str = "float32"
    batch_sizes: Sequence[int] = field(default_factory=lambda: (1, 32, 128, 512, 2048))
    warmup_iters: int = 50
    measure_iters: int = 200
    seq_len: int = 256
    seed: int = 1337
    out_dir: str = "extras/runs/latency/bench"


def _build_algo(cfg: LatencyBenchConfig) -> MAPFGPTInference:
    return MAPFGPTInference(
        MAPFGPTInferenceConfig(
            path_to_weights=cfg.weights,
            device=cfg.device,
            infer_dtype=cfg.dtype,
        )
    )


def _measure(model_act, idx: torch.Tensor, iters: int) -> List[float]:
    times: List[float] = []
    device = idx.device
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            starter.record()
            _ = model_act(idx)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model_act(idx)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)  # ms
    return times


def _stats(samples: List[float]) -> Dict[str, float]:
    s = sorted(samples)
    n = len(s)
    p50 = statistics.median(s)
    p95 = s[int(0.95 * (n - 1))] if n else float("nan")
    mean = statistics.fmean(s) if s else float("nan")
    stdev = statistics.stdev(s) if n >= 2 else 0.0
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": mean, "std_ms": stdev}


def bench_one(cfg: LatencyBenchConfig, repo_root: Path) -> Path:
    """Run a sweep for one checkpoint and append rows to ``latency.csv``."""
    run_dir = make_run_dir(cfg.out_dir, tag=cfg.label.replace(" ", "_") or None)
    write_run_config(run_dir, asdict(cfg))
    setup_run_logging(run_dir, repo_root)
    csv_path = run_dir / "latency.csv"
    cols = ["label", "device", "dtype", "batch_size", "iters", "p50_ms", "p95_ms", "mean_ms", "std_ms", "tokens_per_s"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    torch.manual_seed(cfg.seed)
    algo = _build_algo(cfg)
    device = torch.device(algo.cfg.device)
    model = algo.net

    def model_act(idx: torch.Tensor):
        return model.act(idx, generator=algo.torch_generator)

    rows: List[Dict[str, Any]] = []
    for B in cfg.batch_sizes:
        if cfg.seq_len > model.config.block_size:
            raise ValueError(f"seq_len {cfg.seq_len} exceeds model block_size {model.config.block_size}")
        idx = torch.randint(
            low=0, high=int(model.config.vocab_size),
            size=(B, cfg.seq_len), dtype=torch.long, device=device,
        )
        _measure(model_act, idx, iters=cfg.warmup_iters)
        samples = _measure(model_act, idx, iters=cfg.measure_iters)
        stats = _stats(samples)
        tokens_per_s = B * cfg.seq_len / (stats["mean_ms"] * 1e-3) if stats["mean_ms"] > 0 else 0.0
        row: Dict[str, Any] = {
            "label": cfg.label or Path(cfg.weights).name,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "batch_size": B,
            "iters": cfg.measure_iters,
            "tokens_per_s": tokens_per_s,
            **stats,
        }
        rows.append(row)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writerow(row)
        print(json.dumps(row), flush=True)

    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> LatencyBenchConfig:
    p = argparse.ArgumentParser(description="Pure forward-pass latency benchmark for MAPF-GPT.")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    p.add_argument("--label", default="", help="Human-readable label for CSV rows")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128, 512, 2048])
    p.add_argument("--warmup-iters", type=int, default=50)
    p.add_argument("--measure-iters", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="extras/runs/latency/bench")
    ns = p.parse_args(argv)
    return LatencyBenchConfig(
        weights=ns.weights, label=ns.label, device=ns.device, dtype=ns.dtype,
        batch_sizes=tuple(ns.batch_sizes), warmup_iters=ns.warmup_iters,
        measure_iters=ns.measure_iters, seq_len=ns.seq_len, seed=ns.seed,
        out_dir=ns.out_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    run_dir = bench_one(cfg, PROJECT_ROOT)
    print(f"latency results: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
