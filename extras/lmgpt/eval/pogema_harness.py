"""POGEMA benchmark harness for MAPF-GPT (and any compatible inference adapter).

Produces a streaming JSONL of per-episode metrics + aggregated CSV/MD summaries
+ reproducibility metadata, under ``<output-root>/<timestamp>/`` (defaults to
``extras/runs/baselines/<timestamp>/``).

Originally lived at ``extras/run_author_baselines.py``; consolidated here so
all evaluation tooling sits inside the ``lmgpt`` package.

Run via the ``extras/scripts/eval_pogema.py`` thin wrapper, or import
:func:`main` directly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
# Repo layout: MAPF-GPT/extras/lmgpt/eval/pogema_harness.py
PROJECT_ROOT = SCRIPT_PATH.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXTRAS_ROOT = SCRIPT_PATH.parents[2]
if str(EXTRAS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRAS_ROOT))
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from pogema_toolbox.config_variant_generator import generate_variants
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry

# Repo imports (cwd set to project root in main)
from create_env import create_eval_env
from mapf_gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

DEFAULT_GROUP_ORDER = (
    "01-random",
    "02-mazes",
    "03-warehouse",
    "04-movingai",
    "05-puzzles",
)

MODEL_CHECKPOINTS: Dict[str, str] = {
    "2M": "weights/model-2M.pt",
    "6M": "weights/model-6M.pt",
    "85M": "weights/model-85M.pt",
}

ENV_CFG_NAME = "Environment"
BASE_PATH = Path("eval_configs")

# ---------------------------------------------------------------------------#
# JSON / metrics helpers
# ---------------------------------------------------------------------------#


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        if isinstance(obj, np.floating):
            return float(obj)
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def simplify_grid_changes(change_dict: Mapping) -> Dict[str, Any]:
    return {key[-1]: value for key, value in change_dict.items()}


def z95() -> float:
    return 1.96


def binomial_proportion_ci(successes: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0 or successes is None:
        return None, None
    p = successes / n
    if n <= 1:
        return p, p
    z = z95()
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return lo, hi


def mean_std_sem_ci(values: List[float]) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    arr = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(arr)
    if n == 0:
        return None, None, None, None, None
    mean = sum(arr) / n
    if n == 1:
        return mean, 0.0, None, mean, mean
    variance = sum((x - mean) ** 2 for x in arr) / (n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    half = z95() * sem
    return mean, std, sem, mean - half, mean + half


def median_from_sorted(sorted_vals: List[float]) -> Optional[float]:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


# ---------------------------------------------------------------------------#
# Discovery & configs
# ---------------------------------------------------------------------------#


def discover_groups(repo_root: Path) -> List[str]:
    base = repo_root / BASE_PATH
    found = []
    if not base.is_dir():
        return []
    for name in DEFAULT_GROUP_ORDER:
        if (base / name).is_dir() and (base / name / "maps.yaml").is_file():
            cfg = base / name / f"{name}.yaml"
            if cfg.is_file():
                found.append(name)
    return found


def eval_config_path(group: str) -> Path:
    return PROJECT_ROOT / BASE_PATH / group / f"{group}.yaml"


def load_evaluation_yaml(group: str) -> Dict[str, Any]:
    path = eval_config_path(group)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_results_views(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    if "results_views" in out:
        del out["results_views"]
    return out


def build_algorithm_block(
    model_key: str,
    weights_relpath: str,
    device: str,
    infer_dtype: str,
    num_process: int,
    algo_entry_name: Optional[str] = None,
) -> Dict[str, Any]:
    entry = algo_entry_name or f"MAPF-GPT-{model_key}-author"
    return {
        entry: {
            "name": "MAPF-GPT",
            "parallel_backend": "sequential",
            "num_process": num_process,
            "path_to_weights": weights_relpath,
            "device": device,
            "infer_dtype": infer_dtype,
        }
    }


def apply_quick_subset(env_spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = copy.deepcopy(env_spec)
    for key in ("map_name", "num_agents", "seed"):
        val = spec.get(key)
        if isinstance(val, dict) and "grid_search" in val:
            gs = val["grid_search"]
            if isinstance(gs, list) and gs:
                spec[key] = {"grid_search": [gs[0]]}
    return spec


def register_project_maps(repo_root: Path, groups: Sequence[str]) -> None:
    for group in groups:
        maps_path = repo_root / BASE_PATH / group / "maps.yaml"
        with open(maps_path, "r", encoding="utf-8") as f:
            ToolboxRegistry.register_maps(yaml.safe_load(f))


def setup_registry() -> None:
    ToolboxRegistry.register_env(ENV_CFG_NAME, create_eval_env, Environment)
    ToolboxRegistry.register_algorithm("MAPF-GPT", MAPFGPTInference, MAPFGPTInferenceConfig)


# ---------------------------------------------------------------------------#
# Episode loop (streaming JSONL)
# ---------------------------------------------------------------------------#


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def build_record(
    *,
    model_name: str,
    map_group: str,
    algorithm_display: str,
    metrics: Dict[str, Any],
    env_config: Dict[str, Any],
    env_changes: Dict[str, Any],
    device: str,
    dtype: str,
    checkpoint_path: Path,
    episode_ok: bool,
    error: Optional[str],
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()
    raw_metrics = json_safe(copy.deepcopy(metrics)) if metrics is not None else {}
    runtime_raw = None
    if isinstance(raw_metrics, dict):
        runtime_raw = raw_metrics.get("runtime")

    record: Dict[str, Any] = {
        "model_name": model_name,
        "map_group": map_group,
        "map_name": env_changes.get("map_name"),
        "num_agents": env_changes.get("num_agents"),
        "seed": env_changes.get("seed"),
        "max_episode_steps": env_config.get("max_episode_steps"),
        "ISR": raw_metrics.get("ISR") if isinstance(raw_metrics, dict) else None,
        "CSR": raw_metrics.get("CSR") if isinstance(raw_metrics, dict) else None,
        "SR": raw_metrics.get("CSR") if isinstance(raw_metrics, dict) else None,
        "SoC": raw_metrics.get("SoC") if isinstance(raw_metrics, dict) else None,
        "ep_length": raw_metrics.get("ep_length") if isinstance(raw_metrics, dict) else None,
        "runtime": runtime_raw,
        "episode_runtime_end_to_end": runtime_raw,
        "device": device,
        "dtype": dtype,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": ts,
        "raw_metrics": raw_metrics if isinstance(raw_metrics, dict) else {},
        "algorithm": algorithm_display,
        "episode_ok": episode_ok,
    }
    if error:
        record["error"] = error
    return record


# ---------------------------------------------------------------------------#
# Environment logging
# ---------------------------------------------------------------------------#


def run_cmd_text(cmd: List[str], timeout: int = 30) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    except Exception as exc:  # noqa: BLE001
        return f"(failed to run {cmd}: {exc})"


def collect_environment_text(repo_root: Path) -> str:
    lines: List[str] = []
    lines.append(f"run_timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"hostname: {run_cmd_text(['hostname']).strip()}")
    lines.append(f"cwd: {os.getcwd()}")
    lines.append(f"repo_root: {repo_root}")

    lines.append("\n## git\n")
    lines.append(f"branch: {run_cmd_text(['git', '-C', str(repo_root), 'branch', '--show-current']).strip()}")
    lines.append(f"commit: {run_cmd_text(['git', '-C', str(repo_root), 'rev-parse', 'HEAD']).strip()}")
    lines.append(f"short: {run_cmd_text(['git', '-C', str(repo_root), 'rev-parse', '--short', 'HEAD']).strip()}")

    lines.append("\n## python\n")
    lines.append(run_cmd_text([sys.executable, "--version"]).strip())

    py_snippet = (
        "import torch, numpy as np;\n"
        "import importlib.metadata as im;\n"
        "def ver(*names):\n"
        "    for n in names:\n"
        "        try: return f'{n}=' + im.version(n)\n"
        "        except Exception: pass\n"
        "    return 'not found'\n"
        "print('torch', torch.__version__);\n"
        "print('cuda_available', torch.cuda.is_available());\n"
        "print('cuda_version_torch', getattr(torch.version, 'cuda', None));\n"
        "print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);\n"
        "print('numpy', np.__version__);\n"
        "print(ver('pogema'));\n"
        "print(ver('pogema-toolbox', 'pogema_toolbox'));\n"
    )
    lines.append(run_cmd_text([sys.executable, "-c", py_snippet]).strip())

    lines.append("\n## nvidia-smi\n")
    lines.append(run_cmd_text(["nvidia-smi"], timeout=20))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------#
# Aggregation
# ---------------------------------------------------------------------------#


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Group by (model_name, map_group, num_agents)."""
    stats_err: Dict[str, int] = defaultdict(int)
    groups: Dict[Tuple[str, str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        mk = r.get("model_name")
        g = r.get("map_group")
        na = r.get("num_agents")
        if r.get("episode_ok") is False:
            stats_err["failed_episodes"] += 1
        groups[(mk, g, na)].append(r)

    summaries: List[Dict[str, Any]] = []
    for (model_name, map_group, num_agents), g_rows in sorted(groups.items()):
        ok_rows = [x for x in g_rows if x.get("episode_ok", True) and not x.get("error")]
        n_all = len(g_rows)
        n_ok = len(ok_rows)

        csr_vals = []
        isr_vals = []
        sr_vals = []
        soc_vals = []
        soc_succ = []
        ep_len_vals = []
        rt_vals = []

        for x in ok_rows:
            m = x.get("raw_metrics") or {}
            csr = m.get("CSR")
            if csr is not None:
                csr_vals.append(float(csr))
            isr = m.get("ISR")
            if isr is not None:
                isr_vals.append(float(isr))
            if csr is not None:
                sr_vals.append(float(csr))
            soc = m.get("SoC")
            if soc is not None:
                soc_vals.append(float(soc))
                if csr == 1 or csr == 1.0:
                    soc_succ.append(float(soc))
            el = m.get("ep_length")
            if el is not None:
                ep_len_vals.append(float(el))
            rt = x.get("episode_runtime_end_to_end")
            if rt is None:
                rt = x.get("runtime")
            if rt is not None:
                rt_vals.append(float(rt))

        successes = sum(1 for v in csr_vals if v >= 1.0 - 1e-9)

        csr_mean = sum(csr_vals) / len(csr_vals) if csr_vals else None
        csr_lo, csr_hi = (
            binomial_proportion_ci(successes, len(csr_vals)) if csr_vals else (None, None)
        )

        isr_mean, _, _, isr_lo, isr_hi = (
            mean_std_sem_ci(isr_vals) if isr_vals else (None, None, None, None, None)
        )

        sr_mean = sum(sr_vals) / len(sr_vals) if sr_vals else None
        sr_lo, sr_hi = (
            binomial_proportion_ci(successes, len(sr_vals)) if sr_vals else (None, None)
        )

        soc_mean = sum(soc_vals) / len(soc_vals) if soc_vals else None
        soc_succ_mean = sum(soc_succ) / len(soc_succ) if soc_succ else None

        ep_mean, _, _, _, _ = mean_std_sem_ci(ep_len_vals) if ep_len_vals else (None, None, None, None, None)

        rt_sorted = sorted(rt_vals)
        rt_mean, rt_std, _, rt_lo, rt_hi = (
            mean_std_sem_ci(rt_vals) if rt_vals else (None, None, None, None, None)
        )
        rt_med = median_from_sorted(rt_sorted)

        summaries.append(
            {
                "model_name": model_name,
                "map_group": map_group,
                "num_agents": num_agents,
                "episodes_count": n_all,
                "episodes_ok": n_ok,
                "CSR_mean": csr_mean,
                "CSR_ci95_low": csr_lo,
                "CSR_ci95_high": csr_hi,
                "ISR_mean": isr_mean,
                "ISR_ci95_low": isr_lo,
                "ISR_ci95_high": isr_hi,
                "success_rate_mean": sr_mean,
                "SR_ci95_low": sr_lo,
                "SR_ci95_high": sr_hi,
                "SoC_mean": soc_mean,
                "SoC_mean_success_only": soc_succ_mean,
                "ep_length_mean": ep_mean,
                "runtime_mean": rt_mean,
                "runtime_median": rt_med,
                "runtime_std": rt_std,
                "runtime_ci95_low": rt_lo,
                "runtime_ci95_high": rt_hi,
            }
        )

    return summaries, dict(stats_err)


def write_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(summaries[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summaries:
            w.writerow(row)


def write_summary_md(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        path.write_text("# Summary\n\n(no rows)\n", encoding="utf-8")
        return
    cols = list(summaries[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = ["# Summary by model / map_group / num_agents\n", header, sep]
    for row in summaries:
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_readme(
    run_dir: Path,
    args_ns: argparse.Namespace,
    groups: List[str],
    models: List[str],
    had_errors: bool,
    stats_err: Mapping[str, int],
) -> None:
    cmd = " ".join(sys.argv)
    body = f"""# Author baseline run

## Command

```bash
{cmd}
```

## Models

{", ".join(models)}

## Benchmark groups

{", ".join(groups)}

## Outputs

- `raw_results.jsonl` — one JSON object per line (safe to tail while running).
- `summary_by_model_map_agents.csv` / `.md` — aggregated metrics.
- `run_config.json` — CLI and resolved paths.
- `environment.txt` — git, Python, torch, GPU, nvidia-smi.
- `errors.log` — failures that did not stop the run.

## Notes

- **`episode_runtime_end_to_end`** in each row is POGEMA `runtime` (policy + env stepping), not pure model forward time.
- Interrupted runs: already-written lines in `raw_results.jsonl` remain valid JSON; re-run aggregation by executing the full script again on the same directory is not supported — use completed JSONL for analysis.

## Status

- Errors logged: **{"yes" if had_errors else "no"}**
- Failed episodes (from rows): {stats_err.get("failed_episodes", 0)}
"""
    (run_dir / "README.md").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------#
# Main
# ---------------------------------------------------------------------------#


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Author MAPF-GPT baselines on POGEMA eval_configs (streaming JSONL)."
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan only; do not execute.")
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Subset env grid (first map, first num_agents, first seed where applicable); "
            "only the first discovered benchmark group (stable order)."
        ),
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CHECKPOINTS.keys()),
        default=["2M", "6M", "85M"],
        help="Model sizes (default: 2M 6M 85M).",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda).")
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Maps to MAPFGPT infer_dtype (default: float32).",
    )
    p.add_argument(
        "--num-process",
        type=int,
        default=4,
        help="Stored in YAML algorithm block; streaming uses sequential backend (default: 4).",
    )
    p.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Subset of eval_config groups (default: all discovered in stable order).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="extras/runs/baselines",
        help="Directory under repo root for timestamped runs (default: extras/runs/baselines).",
    )
    p.add_argument(
        "--custom-weights",
        type=str,
        default=None,
        help="Path to a custom .pt checkpoint (e.g. distilled student). Use with --custom-model-name.",
    )
    p.add_argument(
        "--custom-model-name",
        type=str,
        default=None,
        help="Label for raw_results / logs when using --custom-weights (required with that flag).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = PROJECT_ROOT
    os.chdir(repo_root)

    discovered = discover_groups(repo_root)
    if args.groups:
        groups = [g for g in DEFAULT_GROUP_ORDER if g in args.groups and g in discovered]
        missing = set(args.groups) - set(discovered)
        if missing:
            print(f"WARNING: unknown or missing groups (skipped): {sorted(missing)}", file=sys.stderr)
    else:
        groups = discovered

    if args.custom_weights or args.custom_model_name:
        if not args.custom_weights or not args.custom_model_name:
            print(
                "ERROR: --custom-weights and --custom-model-name must be set together.",
                file=sys.stderr,
            )
            return 2
        models = [args.custom_model_name]
    else:
        models = list(args.models)

    run_groups: List[str] = [groups[0]] if (args.quick and groups) else groups

    print("MAPF-GPT author baseline runner")
    print(f"  repo root: {repo_root}")
    print(f"  models: {models}")
    print(f"  groups (all): {groups}")
    print(f"  groups (this run): {run_groups}")
    print(f"  device: {args.device}, dtype: {args.dtype}")
    if args.quick:
        print("  quick mode: subset env grid + only first benchmark group in list")

    if args.dry_run:
        print("\n--- dry-run: planned jobs ---")
        for m in models:
            if args.custom_weights:
                w = str(Path(args.custom_weights).resolve())
            else:
                w = MODEL_CHECKPOINTS[m]
            print(f"model {m}: {w}")
        for g in run_groups:
            cfg = load_evaluation_yaml(g)
            env = cfg["environment"]
            if args.quick:
                env = apply_quick_subset(env)
            n_variants = len(list(generate_variants(env)))
            print(f"  group {g}: {eval_config_path(g)} -> {n_variants} env variants")
        print("\ndry-run complete (no execution).")
        return 0

    if not run_groups:
        print("ERROR: no benchmark groups found under eval_configs/.", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = repo_root / args.output_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "argv": sys.argv,
        "models": models,
        "groups_all_discovered": groups,
        "groups_this_run": run_groups,
        "device": args.device,
        "dtype": args.dtype,
        "infer_dtype": args.dtype,
        "quick": args.quick,
        "output_dir": str(run_dir.relative_to(repo_root)),
        "model_checkpoints": (
            {m: str(Path(args.custom_weights).resolve()) for m in models}
            if args.custom_weights
            else {m: MODEL_CHECKPOINTS[m] for m in models}
        ),
        "custom_weights": args.custom_weights,
        "custom_model_name": args.custom_model_name,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    env_text = collect_environment_text(repo_root)
    (run_dir / "environment.txt").write_text(env_text, encoding="utf-8")

    raw_path = run_dir / "raw_results.jsonl"
    err_path = run_dir / "errors.log"
    err_path.write_text("", encoding="utf-8")
    had_errors = False

    def log_err(msg: str) -> None:
        nonlocal had_errors
        had_errors = True
        with open(err_path, "a", encoding="utf-8") as ef:
            ef.write(msg.rstrip() + "\n")
            ef.flush()

    setup_registry()
    if groups:
        register_project_maps(repo_root, groups)

    for model_key in models:
        if args.custom_weights:
            weights_rel = str(Path(args.custom_weights).resolve())
            ck_abs = Path(weights_rel)
            algo_name_display = model_key
            entry_name = model_key
        else:
            weights_rel = MODEL_CHECKPOINTS[model_key]
            ck_abs = (repo_root / weights_rel).resolve()
            algo_name_display = f"MAPF-GPT-{model_key}-author"
            entry_name = algo_name_display
        algo_block = build_algorithm_block(
            model_key,
            weights_rel,
            args.device,
            args.dtype,
            args.num_process,
            algo_entry_name=entry_name,
        )[entry_name]
        algo_impl_name = algo_block["name"]
        algo = ToolboxRegistry.create_algorithm(algo_impl_name, **algo_block)
        algo_cfg_obj = ToolboxRegistry.create_algorithm_config(algo_impl_name, **algo_block)

        for map_group in run_groups:
            try:
                base_cfg = load_evaluation_yaml(map_group)
                base_cfg = strip_results_views(base_cfg)
                env_block = base_cfg["environment"]
                if args.quick:
                    env_block = apply_quick_subset(env_block)

                pairs = list(generate_variants(env_block))
                env_changes_list = [p[0] for p in pairs]
                env_configs = [p[1] for p in pairs]

                for idx, env_config in enumerate(env_configs):
                    try:
                        ToolboxRegistry.info(
                            f"Running: {algo_name_display} [{idx + 1}/{len(env_configs)}]"
                        )
                        env = ToolboxRegistry.create_env(env_config["name"], **env_config)
                        if algo_cfg_obj.preprocessing:
                            env = ToolboxRegistry.create_algorithm_preprocessing(
                                env, algo_impl_name, **algo_block
                            )
                        metric = ToolboxRegistry.run_episode(
                            env, algo, algo_cfg_obj.run_episode_func
                        )
                        ch = simplify_grid_changes(env_changes_list[idx])
                        rec = build_record(
                            model_name=model_key,
                            map_group=map_group,
                            algorithm_display=algo_name_display,
                            metrics=metric,
                            env_config=env_config,
                            env_changes=ch,
                            device=args.device,
                            dtype=args.dtype,
                            checkpoint_path=ck_abs,
                            episode_ok=True,
                            error=None,
                        )
                        append_jsonl(raw_path, rec)
                    except Exception:
                        tb = traceback.format_exc()
                        log_err(f"[{model_key}][{map_group}] episode {idx}:\n{tb}")
                        ch = simplify_grid_changes(env_changes_list[idx])
                        rec = build_record(
                            model_name=model_key,
                            map_group=map_group,
                            algorithm_display=algo_name_display,
                            metrics={},
                            env_config=env_configs[idx],
                            env_changes=ch,
                            device=args.device,
                            dtype=args.dtype,
                            checkpoint_path=ck_abs,
                            episode_ok=False,
                            error=tb,
                        )
                        append_jsonl(raw_path, rec)

            except Exception:
                tb = traceback.format_exc()
                log_err(f"[{model_key}][{map_group}] group failure:\n{tb}")
                continue

    rows = load_jsonl(raw_path)
    summaries, stats_err = aggregate(rows)
    write_summary_csv(run_dir / "summary_by_model_map_agents.csv", summaries)
    write_summary_md(run_dir / "summary_by_model_map_agents.md", summaries)

    write_run_readme(run_dir, args, run_groups, models, had_errors, stats_err)

    # Refresh environment.txt with end-of-run timestamp
    (run_dir / "environment.txt").write_text(collect_environment_text(repo_root), encoding="utf-8")

    print(f"\nDone. Results in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
