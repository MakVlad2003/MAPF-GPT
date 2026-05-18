#!/usr/bin/env python3
"""
Aggregate MAPF-GPT benchmark outputs into comparison tables (baseline vs future runs).

Reads one or more result directories containing ``raw_results.jsonl`` (preferred) or
``summary_by_model_map_agents.csv``. Does not run benchmarks.

**SoC:** By default ``SoC_mean`` follows full-episode averaging (all episodes with SoC),
while ``SoC_on_common_success_with_baseline`` averages only on episode keys where both the
model and the chosen baseline CSR-succeed (requires ``map_name`` in raw JSONL for alignment).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Aligns episodes across models within a run (requires map_name in raw JSONL).
EpisodeKey = Tuple[str, str, int, str]

SCRIPT_PATH = Path(__file__).resolve()
# Repo layout: MAPF-GPT/extras/lmgpt/eval/compare_runs.py
PROJECT_ROOT = SCRIPT_PATH.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

Z95 = 1.959963984540054


def wilson_interval(successes: float, n: int) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion; successes can be float sum of 0/1."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    z = Z95
    zz = z * z
    denom = 1.0 + zz / n
    center = (p + zz / (2.0 * n)) / denom
    rad = z * math.sqrt((p * (1.0 - p) + zz / (4.0 * n)) / n) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def safe_float(x: Any, default: float = float("nan")) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def extract_soc(row: Dict[str, Any], raw_metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    for k in ("SoC", "soc", "sum_of_costs"):
        if k in row and row[k] is not None:
            return safe_float(row[k], float("nan"))
    if raw_metrics:
        for k in ("SoC", "soc"):
            if k in raw_metrics and raw_metrics[k] is not None:
                return safe_float(raw_metrics[k], float("nan"))
    return None


def extract_runtime(row: Dict[str, Any]) -> Optional[float]:
    for k in ("episode_runtime_end_to_end", "runtime", "episode_runtime"):
        if k in row and row[k] is not None:
            v = safe_float(row[k], float("nan"))
            if not math.isnan(v):
                return v
    return None


def normalize_episode_row(obj: Dict[str, Any], source_run: str) -> Optional[Dict[str, Any]]:
    """Build a flat episode dict; return None if unusable."""
    model = obj.get("model_name") or obj.get("model")
    if not model:
        return None
    model = str(model).strip()
    map_group = obj.get("map_group")
    if not map_group:
        return None
    map_group = str(map_group).strip()
    num_agents = obj.get("num_agents")
    if num_agents is None:
        return None
    try:
        num_agents = int(num_agents)
    except (TypeError, ValueError):
        return None

    raw_metrics = obj.get("raw_metrics")
    if isinstance(raw_metrics, str):
        try:
            raw_metrics = json.loads(raw_metrics)
        except json.JSONDecodeError:
            raw_metrics = None
    if raw_metrics is not None and not isinstance(raw_metrics, dict):
        raw_metrics = None

    csr = obj.get("CSR")
    if csr is None:
        csr = obj.get("success_rate") or obj.get("success_rate_mean")
    if csr is None and raw_metrics:
        csr = raw_metrics.get("CSR")
    csr = safe_float(csr, float("nan"))

    sr = obj.get("SR") or obj.get("success_rate")
    if sr is None:
        sr = csr
    sr = safe_float(sr, float("nan"))

    isr = obj.get("ISR")
    if isr is None and raw_metrics:
        isr = raw_metrics.get("ISR")
    isr = safe_float(isr, float("nan"))

    soc = extract_soc(obj, raw_metrics)
    ep_len = obj.get("ep_length")
    if ep_len is None and raw_metrics:
        ep_len = raw_metrics.get("ep_length")
    ep_len = safe_float(ep_len, float("nan")) if ep_len is not None else float("nan")

    rt = extract_runtime(obj)
    if rt is None:
        rt = float("nan")

    map_name_raw = obj.get("map_name")
    map_name = str(map_name_raw) if map_name_raw is not None else ""

    return {
        "model_name": model,
        "map_group": map_group,
        "map_name": map_name,
        "num_agents": num_agents,
        "CSR": csr,
        "ISR": isr,
        "SR": sr,
        "SoC": soc,
        "ep_length": ep_len,
        "runtime": rt,
        "source_run": source_run,
        "seed": obj.get("seed"),
    }


def is_success_episode(csr: float) -> bool:
    if math.isnan(csr):
        return False
    if csr >= 1.0 - 1e-9:
        return True
    if csr <= 0.0 + 1e-9:
        return False
    return csr >= 0.5


def load_jsonl_episodes(path: Path, source_run: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = normalize_episode_row(obj, source_run)
            if row:
                out.append(row)
    return out


def load_summary_csv_episodes(
    path: Path, source_run: str
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Expand summary buckets into pseudo-episodes by repeating bucket means (for weighting).
    Returns (episodes, used_fallback_notice).
    """
    out: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row["episodes_count"])
            except (KeyError, ValueError):
                continue
            model = (row.get("model_name") or "").strip()
            g = (row.get("map_group") or "").strip()
            if not model or not g:
                continue
            try:
                na = int(row["num_agents"])
            except (KeyError, ValueError):
                continue
            csr = safe_float(row.get("CSR_mean"), float("nan"))
            isr = safe_float(row.get("ISR_mean"), float("nan"))
            sr = safe_float(row.get("success_rate_mean"), csr)
            soc_m = safe_float(row.get("SoC_mean"), float("nan"))
            soc_s = safe_float(row.get("SoC_mean_success_only"), float("nan"))
            ep_m = safe_float(row.get("ep_length_mean"), float("nan"))
            rt_m = safe_float(row.get("runtime_mean"), float("nan"))
            # Synthetic: one row per bucket carrying mean values (weight = n)
            for _ in range(n):
                out.append(
                    {
                        "model_name": model,
                        "map_group": g,
                        "map_name": "",
                        "num_agents": na,
                        "CSR": csr,
                        "ISR": isr,
                        "SR": sr,
                        "SoC": soc_m,
                        "SoC_success_bucket": soc_s,
                        "ep_length": ep_m,
                        "runtime": rt_m,
                        "source_run": source_run,
                        "seed": None,
                        "_from_summary": True,
                    }
                )
    return out, True


def load_run_directory(
    run_dir: Path,
    raw_name: str,
    summary_name: str,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], str, Optional[Path]]:
    """Returns episodes, source_run label, path to raw if used."""
    raw_path = run_dir / raw_name
    summary_path = run_dir / summary_name
    label = run_dir.name
    if raw_path.is_file():
        eps = load_jsonl_episodes(raw_path, label)
        if verbose:
            print(f"[{label}] loaded {len(eps)} episodes from {raw_path}", file=sys.stderr)
        return eps, label, raw_path
    if summary_path.is_file():
        eps, _ = load_summary_csv_episodes(summary_path, label)
        if verbose:
            print(
                f"[{label}] loaded {len(eps)} synthetic rows from {summary_path} (no raw)",
                file=sys.stderr,
            )
        return eps, label, None
    raise FileNotFoundError(f"No {raw_name} or {summary_name} in {run_dir}")


def read_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "run_config.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def dtype_to_precision(dtype: Optional[str]) -> str:
    if not dtype:
        return ""
    d = str(dtype).lower()
    if d in ("float32", "fp32", "32"):
        return "FP32"
    if d in ("bfloat16", "bf16"):
        return "BF16"
    if d in ("float16", "fp16", "16"):
        return "FP16"
    if "int8" in d or d == "int8":
        return "INT8"
    return dtype.upper()


def infer_params_from_model(name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Mm]\b", name)
    if m:
        return m.group(1) + "M"
    m = re.search(r"\b(2M|6M|85M|0\.5M|1M|1\.5M)\b", name, re.I)
    if m:
        return m.group(1)
    return ""


def infer_method(run_dir: Path, model_name: str, run_cfg: Optional[Dict[str, Any]]) -> str:
    s = run_dir.as_posix().lower()
    if "distill" in s:
        return "distillation"
    if "quant" in s or "int8" in s:
        return "quantization"
    if "trim" in s:
        return "trim"
    if "baseline" in s:
        return "author_baseline"
    if run_cfg and run_cfg.get("argv"):
        argv = " ".join(str(x) for x in run_cfg["argv"])
        if "custom-weights" in argv:
            return "custom_eval"
    return ""


def filter_episodes(
    episodes: List[Dict[str, Any]],
    drop_puzzles: bool,
    only_groups: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    out = []
    for e in episodes:
        g = e["map_group"]
        if drop_puzzles and g == "05-puzzles":
            continue
        if only_groups is not None and len(only_groups) > 0 and g not in only_groups:
            continue
        out.append(e)
    return out


def soc_success_value(ep: Dict[str, Any]) -> float:
    if ep.get("_from_summary") and ep.get("SoC_success_bucket") is not None:
        v = ep["SoC_success_bucket"]
        if not math.isnan(safe_float(v)):
            return safe_float(v)
    s = ep.get("SoC")
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return float("nan")
    return float(s)


def aggregate_group(
    eps: Sequence[Dict[str, Any]],
    *,
    include_failed_soc: bool = True,
) -> Dict[str, Any]:
    """Single aggregation over a list of episodes.

    If ``include_failed_soc`` is False (CLI default), ``SoC_mean`` uses only successful
    episodes; if True, ``SoC_mean`` averages SoC over all episodes with valid SoC.
    ``SoC_success_only_mean`` is always successful episodes only.
    """
    n = len(eps)
    if n == 0:
        return {
            "episodes": 0,
            "CSR_mean": float("nan"),
            "ISR_mean": float("nan"),
            "SR_mean": float("nan"),
            "SoC_mean": float("nan"),
            "SoC_success_only_mean": float("nan"),
            "ep_length_mean": float("nan"),
            "runtime_mean": float("nan"),
            "runtime_median": float("nan"),
            "runtime_std": float("nan"),
            "wilson_CSR": (float("nan"), float("nan")),
            "wilson_ISR": (float("nan"), float("nan")),
            "wilson_SR": (float("nan"), float("nan")),
        }

    csrs = [e["CSR"] for e in eps if not math.isnan(e["CSR"])]
    isrs = [e["ISR"] for e in eps if not math.isnan(e["ISR"])]
    srs = [e["SR"] for e in eps if not math.isnan(e["SR"])]

    csr_mean = statistics.mean(csrs) if csrs else float("nan")
    isr_mean = statistics.mean(isrs) if isrs else float("nan")
    sr_mean = statistics.mean(srs) if srs else float("nan")

    soc_all_vals: List[float] = []
    for e in eps:
        s = e.get("SoC")
        if s is None:
            continue
        sf = safe_float(s, float("nan"))
        if not math.isnan(sf):
            soc_all_vals.append(sf)

    succ_eps = [e for e in eps if is_success_episode(e["CSR"])]
    soc_succ = [
        soc_success_value(e)
        for e in succ_eps
        if not math.isnan(soc_success_value(e))
    ]

    soc_only = statistics.mean(soc_succ) if soc_succ else float("nan")
    if include_failed_soc:
        soc_mean = statistics.mean(soc_all_vals) if soc_all_vals else float("nan")
    else:
        soc_mean = soc_only if soc_succ else (statistics.mean(soc_all_vals) if soc_all_vals else float("nan"))
    if math.isnan(soc_only) and soc_all_vals and include_failed_soc:
        soc_only = soc_mean

    ep_lens = [e["ep_length"] for e in eps if not math.isnan(e["ep_length"])]
    ep_len_mean = statistics.mean(ep_lens) if ep_lens else float("nan")

    rts = [e["runtime"] for e in eps if not math.isnan(e["runtime"])]
    rt_mean = statistics.mean(rts) if rts else float("nan")
    rt_med = statistics.median(rts) if rts else float("nan")
    rt_std = statistics.stdev(rts) if len(rts) > 1 else 0.0

    n_csr, n_isr, n_sr = len(csrs), len(isrs), len(srs)

    w_csr = wilson_interval(sum(csrs), n_csr) if n_csr else (float("nan"), float("nan"))
    w_isr = wilson_interval(sum(isrs), n_isr) if n_isr else (float("nan"), float("nan"))
    w_sr = wilson_interval(sum(srs), n_sr) if n_sr else (float("nan"), float("nan"))

    return {
        "episodes": n,
        "CSR_mean": csr_mean,
        "ISR_mean": isr_mean,
        "SR_mean": sr_mean,
        "SoC_mean": soc_mean,
        "SoC_success_only_mean": soc_only if not math.isnan(soc_only) else soc_mean,
        "ep_length_mean": ep_len_mean,
        "runtime_mean": rt_mean,
        "runtime_median": rt_med,
        "runtime_std": rt_std,
        "wilson_CSR": w_csr,
        "wilson_ISR": w_isr,
        "wilson_SR": w_sr,
    }


def group_key_global(e: Dict[str, Any]) -> Tuple[str, str]:
    return (e["source_run"], e["model_name"])


def group_key_group(e: Dict[str, Any]) -> Tuple[str, str, str]:
    return (e["source_run"], e["model_name"], e["map_group"])


def group_key_full(e: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (e["source_run"], e["model_name"], e["map_group"], int(e["num_agents"]))


def episode_key_from_ep(e: Dict[str, Any]) -> Optional[EpisodeKey]:
    """Stable key for the same POGEMA episode across models (needs ``map_name`` in raw JSONL)."""
    if e.get("_from_summary"):
        return None
    mn = e.get("map_name")
    if mn is None or str(mn).strip() == "":
        return None
    seed = e.get("seed")
    seed_repr = json.dumps(seed, sort_keys=True)
    return (str(e["map_group"]), str(mn).strip(), int(e["num_agents"]), seed_repr)


def index_episodes_by_key(eps: Sequence[Dict[str, Any]]) -> Dict[EpisodeKey, Dict[str, Any]]:
    out: Dict[EpisodeKey, Dict[str, Any]] = {}
    for e in eps:
        k = episode_key_from_ep(e)
        if k is not None:
            out[k] = e
    return out


def pairwise_common_soc_means(
    baseline_by_key: Dict[EpisodeKey, Dict[str, Any]],
    model_by_key: Dict[EpisodeKey, Dict[str, Any]],
    *,
    key_filter: Optional[Callable[[EpisodeKey], bool]] = None,
) -> Tuple[float, float, int]:
    """
    Mean SoC for model / baseline on episodes where **both** CSR-succeed.
    Returns ``(mean_soc_model, mean_soc_baseline_on_same_episodes, count)``.
    """
    m_soc: List[float] = []
    b_soc: List[float] = []
    for k in baseline_by_key:
        if key_filter is not None and not key_filter(k):
            continue
        if k not in model_by_key:
            continue
        be = baseline_by_key[k]
        me = model_by_key[k]
        if not is_success_episode(be["CSR"]) or not is_success_episode(me["CSR"]):
            continue
        sb = be.get("SoC")
        sm = me.get("SoC")
        if sb is None or sm is None:
            continue
        fb = safe_float(sb, float("nan"))
        fm = safe_float(sm, float("nan"))
        if math.isnan(fb) or math.isnan(fm):
            continue
        m_soc.append(fm)
        b_soc.append(fb)
    if not m_soc:
        return (float("nan"), float("nan"), 0)
    return (statistics.mean(m_soc), statistics.mean(b_soc), len(m_soc))


def build_metadata(
    run_dirs: Sequence[Path],
    episodes: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """(source_run, model_name) -> method, params, precision."""
    meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    run_dir_map = {d.name: d for d in run_dirs}
    for e in episodes:
        key = (e["source_run"], e["model_name"])
        if key in meta:
            continue
        rd = run_dir_map.get(e["source_run"])
        method = params = prec = ""
        if rd:
            cfg = read_run_config(rd)
            prec = dtype_to_precision(cfg.get("dtype") or cfg.get("infer_dtype") if cfg else None)
            method = infer_method(rd, e["model_name"], cfg)
        params = infer_params_from_model(e["model_name"])
        meta[key] = {"method": method, "params": params, "precision": prec}
    return meta


def find_baseline_episodes(
    episodes: List[Dict[str, Any]],
    baseline_model: str,
    first_run_label: str,
) -> List[Dict[str, Any]]:
    bm = baseline_model.strip()
    sub = [e for e in episodes if e["model_name"] == bm and e["source_run"] == first_run_label]
    if sub:
        return sub
    sub = [e for e in episodes if e["model_name"] == bm]
    if sub:
        return sub
    # prefix / contains
    sub = [e for e in episodes if bm in e["model_name"]]
    return sub


def relative_global_row(
    model_row: Dict[str, Any],
    baseline_row: Dict[str, Any],
    baseline_model: str,
) -> Dict[str, Any]:
    def d(name: str) -> float:
        return model_row[name] - baseline_row[name]

    m_rt = model_row["runtime_mean"]
    b_rt = baseline_row["runtime_mean"]
    ratio = m_rt / b_rt if b_rt and not math.isnan(b_rt) and not math.isnan(m_rt) else float("nan")
    speedup = b_rt / m_rt if m_rt and not math.isnan(m_rt) and not math.isnan(b_rt) else float("nan")

    mb = model_row["SoC_success_only_mean"]
    bb = baseline_row["SoC_success_only_mean"]
    ratio_soc = mb / bb if bb and not math.isnan(bb) and not math.isnan(mb) else float("nan")

    csr_m = model_row["CSR_mean"]
    csr_b = baseline_row["CSR_mean"]

    def pass_eps(eps_pct: float) -> bool:
        if math.isnan(csr_m) or math.isnan(csr_b):
            return False
        return csr_m >= csr_b - (eps_pct / 100.0)

    mm_c = safe_float(model_row.get("SoC_on_common_success_with_baseline"), float("nan"))
    bb_c = safe_float(model_row.get("SoC_baseline_on_same_common_episodes"), float("nan"))
    soc_common_delta = mm_c - bb_c if not math.isnan(mm_c) and not math.isnan(bb_c) else float("nan")

    return {
        "model_name": model_row["model_name"],
        "source_run": model_row["source_run"],
        "baseline_model_ref": baseline_model,
        "CSR_delta_vs_baseline": d("CSR_mean"),
        "ISR_delta_vs_baseline": d("ISR_mean"),
        "SR_delta_vs_baseline": d("SR_mean"),
        "SoC_success_only_delta_vs_baseline": d("SoC_success_only_mean"),
        "SoC_success_only_ratio_vs_baseline": ratio_soc,
        "SoC_on_common_success_with_baseline": mm_c,
        "SoC_baseline_on_same_common_episodes": bb_c,
        "SoC_common_pair_episodes_n": model_row.get("SoC_common_pair_episodes_n", 0),
        "SoC_common_delta_pairwise": soc_common_delta,
        "runtime_ratio_vs_baseline": ratio,
        "speedup_vs_baseline": speedup,
        "quality_pass_epsilon_1pct": pass_eps(1.0),
        "quality_pass_epsilon_3pct": pass_eps(3.0),
        "quality_pass_epsilon_5pct": pass_eps(5.0),
    }


def relative_group_row(
    model_row: Dict[str, Any],
    baseline_row: Dict[str, Any],
    map_group: str,
) -> Dict[str, Any]:
    m_rt = model_row["runtime_mean"]
    b_rt = baseline_row["runtime_mean"]
    ratio = m_rt / b_rt if b_rt and not math.isnan(b_rt) and not math.isnan(m_rt) else float("nan")
    speedup = b_rt / m_rt if m_rt and not math.isnan(m_rt) and not math.isnan(b_rt) else float("nan")
    mm_c = safe_float(model_row.get("SoC_on_common_success_with_baseline"), float("nan"))
    bb_c = safe_float(model_row.get("SoC_baseline_on_same_common_episodes"), float("nan"))
    soc_cd = mm_c - bb_c if not math.isnan(mm_c) and not math.isnan(bb_c) else float("nan")
    return {
        "model_name": model_row["model_name"],
        "source_run": model_row["source_run"],
        "map_group": map_group,
        "CSR_delta_vs_baseline": model_row["CSR_mean"] - baseline_row["CSR_mean"],
        "ISR_delta_vs_baseline": model_row["ISR_mean"] - baseline_row["ISR_mean"],
        "SR_delta_vs_baseline": model_row["SR_mean"] - baseline_row["SR_mean"],
        "SoC_success_only_delta_vs_baseline": model_row["SoC_success_only_mean"]
        - baseline_row["SoC_success_only_mean"],
        "SoC_common_delta_pairwise": soc_cd,
        "SoC_common_pair_episodes_n": model_row.get("SoC_common_pair_episodes_n", 0),
        "runtime_ratio_vs_baseline": ratio,
        "speedup_vs_baseline": speedup,
    }


def fmt_md_float(x: float, nd: int) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.{nd}f}"


def fmt_md_speedup(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return ""
    return f"{x:.2f}×"


def write_csv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def md_table(headers: Sequence[str], rows: List[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build comparison tables from baseline/eval result directories."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Result directory paths")
    parser.add_argument("--baseline-model", default="2M", help="Reference model name for deltas")
    parser.add_argument("--out-dir", required=True, help="Output directory for tables")
    parser.add_argument("--raw-filename", default="raw_results.jsonl")
    parser.add_argument("--summary-filename", default="summary_by_model_map_agents.csv")
    parser.add_argument(
        "--drop-puzzles",
        action="store_true",
        help="Exclude map group 05-puzzles",
    )
    parser.add_argument(
        "--only-groups",
        nargs="*",
        default=None,
        help="If set, only include these map_group values",
    )
    parser.add_argument(
        "--soc-mean-success-only",
        action="store_true",
        help="SoC_mean averages only CSR-successful episodes (often equals SoC_success_only_mean). "
        "Default: SoC_mean averages all episodes with valid SoC — same convention as "
        "summary_by_model_map_agents.csv SoC_mean.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    # include_failed_soc True => SoC_mean uses failures too (wider pool)
    include_failed_soc_flag = not args.soc_mean_success_only

    run_dirs = [Path(p).resolve() for p in args.runs]
    for d in run_dirs:
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            return 1

    all_eps: List[Dict[str, Any]] = []
    used_raw = False
    warnings: List[str] = []

    for d in run_dirs:
        eps, label, raw_p = load_run_directory(d, args.raw_filename, args.summary_filename, args.verbose)
        all_eps.extend(eps)
        if raw_p is not None:
            used_raw = True
        if eps and all("_from_summary" in e for e in eps[: min(20, len(eps))]):
            if raw_p is None:
                warnings.append(f"Run `{label}` used summary CSV fallback (no per-episode CI precision).")

    all_eps = filter_episodes(all_eps, args.drop_puzzles, args.only_groups)

    # Success flag / SoC
    no_csr = sum(1 for e in all_eps if math.isnan(e["CSR"]))
    if no_csr:
        warnings.append(f"{no_csr} episodes had missing CSR; success-only SoC may be unreliable.")

    first_label = run_dirs[0].name
    baseline_eps = find_baseline_episodes(all_eps, args.baseline_model, first_label)
    if not baseline_eps:
        print(
            f"No episodes for baseline model `{args.baseline_model}` (run `{first_label}`).",
            file=sys.stderr,
        )
        return 1

    meta = build_metadata(run_dirs, all_eps)

    baseline_by_key = index_episodes_by_key(baseline_eps)
    if len(baseline_by_key) == 0:
        warnings.append(
            "SoC_on_common_success_with_baseline is unavailable: need per-episode ``map_name`` "
            "(use ``raw_results.jsonl`` from ``extras/run_author_baselines.py``, not CSV-only fallback)."
        )

    # --- global_by_model ---
    g_global: Dict[Tuple[str, str], List] = defaultdict(list)
    for e in all_eps:
        g_global[group_key_global(e)].append(e)

    global_rows: List[Dict[str, Any]] = []
    for key in sorted(g_global.keys()):
        src, mname = key
        agg = aggregate_group(g_global[key], include_failed_soc=include_failed_soc_flag)
        md = meta.get(key, {})
        midx = index_episodes_by_key(g_global[key])
        mm_c, bb_c, nn = pairwise_common_soc_means(baseline_by_key, midx)
        global_rows.append(
            {
                "model_name": mname,
                "method": md.get("method", ""),
                "params": md.get("params", ""),
                "precision": md.get("precision", ""),
                "source_run": src,
                **agg,
                "SoC_on_common_success_with_baseline": mm_c,
                "SoC_baseline_on_same_common_episodes": bb_c,
                "SoC_common_pair_episodes_n": nn,
            }
        )

    baseline_row = next(
        (
            r
            for r in global_rows
            if r["model_name"] == args.baseline_model and r["source_run"] == first_label
        ),
        None,
    )
    if baseline_row is None:
        print(
            f"Could not resolve baseline row for `{args.baseline_model}` in `{first_label}`.",
            file=sys.stderr,
        )
        return 1

    # --- by_map_group ---
    g_mg: Dict[Tuple[str, str, str], List] = defaultdict(list)
    for e in all_eps:
        g_mg[group_key_group(e)].append(e)

    by_group_rows: List[Dict[str, Any]] = []
    for key in sorted(g_mg.keys()):
        src, mname, mg = key
        agg = aggregate_group(g_mg[key], include_failed_soc=include_failed_soc_flag)
        midx = index_episodes_by_key(g_mg[key])
        mm_c, bb_c, nn = pairwise_common_soc_means(
            baseline_by_key, midx, key_filter=lambda k, mg=mg: k[0] == mg
        )
        by_group_rows.append(
            {
                "model_name": mname,
                "source_run": src,
                "map_group": mg,
                "episodes": agg["episodes"],
                "CSR_mean": agg["CSR_mean"],
                "ISR_mean": agg["ISR_mean"],
                "SR_mean": agg["SR_mean"],
                "SoC_mean": agg["SoC_mean"],
                "SoC_success_only_mean": agg["SoC_success_only_mean"],
                "SoC_on_common_success_with_baseline": mm_c,
                "SoC_baseline_on_same_common_episodes": bb_c,
                "SoC_common_pair_episodes_n": nn,
                "ep_length_mean": agg["ep_length_mean"],
                "runtime_mean": agg["runtime_mean"],
            }
        )

    # --- by_map_group_agents ---
    g_full: Dict[Tuple[str, str, str, int], List] = defaultdict(list)
    for e in all_eps:
        g_full[group_key_full(e)].append(e)

    by_ga_rows: List[Dict[str, Any]] = []
    for key in sorted(g_full.keys()):
        src, mname, mg, na = key
        agg = aggregate_group(g_full[key], include_failed_soc=include_failed_soc_flag)
        lo_c, hi_c = agg["wilson_CSR"]
        lo_i, hi_i = agg["wilson_ISR"]
        lo_s, hi_s = agg["wilson_SR"]
        midx_ga = index_episodes_by_key(g_full[key])
        mm_cg, bb_cg, nn_g = pairwise_common_soc_means(
            baseline_by_key,
            midx_ga,
            key_filter=lambda k, mg=mg, na=na: k[0] == mg and k[2] == na,
        )
        by_ga_rows.append(
            {
                "model_name": mname,
                "source_run": src,
                "map_group": mg,
                "num_agents": na,
                "episodes": agg["episodes"],
                "CSR_mean": agg["CSR_mean"],
                "ISR_mean": agg["ISR_mean"],
                "SR_mean": agg["SR_mean"],
                "SoC_mean": agg["SoC_mean"],
                "SoC_success_only_mean": agg["SoC_success_only_mean"],
                "SoC_on_common_success_with_baseline": mm_cg,
                "SoC_baseline_on_same_common_episodes": bb_cg,
                "SoC_common_pair_episodes_n": nn_g,
                "ep_length_mean": agg["ep_length_mean"],
                "runtime_mean": agg["runtime_mean"],
                "CSR_ci95_low": lo_c,
                "CSR_ci95_high": hi_c,
                "ISR_ci95_low": lo_i,
                "ISR_ci95_high": hi_i,
                "SR_ci95_low": lo_s,
                "SR_ci95_high": hi_s,
            }
        )

    # --- relative global ---
    rel_global: List[Dict[str, Any]] = []
    for row in global_rows:
        rel_global.append(relative_global_row(row, baseline_row, args.baseline_model))

    # baseline slice per map_group
    def baseline_eps_for_group(mg: str) -> List[Dict[str, Any]]:
        return [e for e in baseline_eps if e["map_group"] == mg]

    rel_by_g: List[Dict[str, Any]] = []
    groups_seen = sorted({e["map_group"] for e in all_eps})
    for mg in groups_seen:
        b_eps_g = baseline_eps_for_group(mg)
        if not b_eps_g:
            continue
        b_agg = aggregate_group(b_eps_g, include_failed_soc=include_failed_soc_flag)
        for key in sorted(g_mg.keys()):
            if key[2] != mg:
                continue
            src_k, mname_k, _ = key
            m_agg = aggregate_group(g_mg[key], include_failed_soc=include_failed_soc_flag)
            m_idx_r = index_episodes_by_key(g_mg[key])
            mm_cr, bb_cr, nn_r = pairwise_common_soc_means(
                baseline_by_key, m_idx_r, key_filter=lambda k, g=mg: k[0] == g
            )
            m_row = {
                "model_name": mname_k,
                "source_run": src_k,
                **m_agg,
                "SoC_on_common_success_with_baseline": mm_cr,
                "SoC_baseline_on_same_common_episodes": bb_cr,
                "SoC_common_pair_episodes_n": nn_r,
            }
            rel_by_g.append(relative_group_row(m_row, b_agg, mg))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    glob_fields = [
        "model_name",
        "method",
        "params",
        "precision",
        "source_run",
        "episodes",
        "CSR_mean",
        "ISR_mean",
        "SR_mean",
        "SoC_mean",
        "SoC_success_only_mean",
        "SoC_on_common_success_with_baseline",
        "SoC_baseline_on_same_common_episodes",
        "SoC_common_pair_episodes_n",
        "ep_length_mean",
        "runtime_mean",
        "runtime_median",
        "runtime_std",
    ]
    write_csv(out_dir / "global_by_model.csv", glob_fields, global_rows)

    mg_fields = [
        "model_name",
        "source_run",
        "map_group",
        "episodes",
        "CSR_mean",
        "ISR_mean",
        "SR_mean",
        "SoC_mean",
        "SoC_success_only_mean",
        "SoC_on_common_success_with_baseline",
        "SoC_baseline_on_same_common_episodes",
        "SoC_common_pair_episodes_n",
        "ep_length_mean",
        "runtime_mean",
    ]
    write_csv(out_dir / "by_map_group.csv", mg_fields, by_group_rows)

    ga_fields = [
        "model_name",
        "source_run",
        "map_group",
        "num_agents",
        "episodes",
        "CSR_mean",
        "ISR_mean",
        "SR_mean",
        "SoC_mean",
        "SoC_success_only_mean",
        "SoC_on_common_success_with_baseline",
        "SoC_baseline_on_same_common_episodes",
        "SoC_common_pair_episodes_n",
        "ep_length_mean",
        "runtime_mean",
        "CSR_ci95_low",
        "CSR_ci95_high",
        "ISR_ci95_low",
        "ISR_ci95_high",
        "SR_ci95_low",
        "SR_ci95_high",
    ]
    write_csv(out_dir / "by_map_group_agents.csv", ga_fields, by_ga_rows)

    rel_g_fields = [
        "model_name",
        "source_run",
        "baseline_model_ref",
        "CSR_delta_vs_baseline",
        "ISR_delta_vs_baseline",
        "SR_delta_vs_baseline",
        "SoC_success_only_delta_vs_baseline",
        "SoC_success_only_ratio_vs_baseline",
        "SoC_on_common_success_with_baseline",
        "SoC_baseline_on_same_common_episodes",
        "SoC_common_pair_episodes_n",
        "SoC_common_delta_pairwise",
        "runtime_ratio_vs_baseline",
        "speedup_vs_baseline",
        "quality_pass_epsilon_1pct",
        "quality_pass_epsilon_3pct",
        "quality_pass_epsilon_5pct",
    ]
    write_csv(out_dir / "relative_to_baseline_global.csv", rel_g_fields, rel_global)

    rel_b_fields = [
        "model_name",
        "source_run",
        "map_group",
        "CSR_delta_vs_baseline",
        "ISR_delta_vs_baseline",
        "SR_delta_vs_baseline",
        "SoC_success_only_delta_vs_baseline",
        "SoC_common_delta_pairwise",
        "SoC_common_pair_episodes_n",
        "runtime_ratio_vs_baseline",
        "speedup_vs_baseline",
    ]
    write_csv(out_dir / "relative_to_baseline_by_group.csv", rel_b_fields, rel_by_g)

    # Markdown
    def row_global_md(r: Dict[str, Any]) -> Tuple:
        return (
            r["model_name"],
            r.get("source_run", ""),
            r.get("method", ""),
            r.get("params", ""),
            r.get("precision", ""),
            fmt_md_float(r["CSR_mean"], 4),
            fmt_md_float(r["ISR_mean"], 4),
            fmt_md_float(r["SR_mean"], 4),
            fmt_md_float(r["SoC_mean"], 2),
            fmt_md_float(r["SoC_success_only_mean"], 2),
            fmt_md_float(r["SoC_on_common_success_with_baseline"], 2),
            fmt_md_float(r["SoC_baseline_on_same_common_episodes"], 2),
            str(r.get("SoC_common_pair_episodes_n", "")),
            fmt_md_float(r["runtime_mean"], 3),
        )

    (out_dir / "global_by_model.md").write_text(
        md_table(
            [
                "model_name",
                "source_run",
                "method",
                "params",
                "prec",
                "CSR",
                "ISR",
                "SR",
                "SoC_mean",
                "SoC_succ",
                "SoC_cm",
                "SoC_bl_cm",
                "n_cm",
                "runtime_s",
            ],
            [row_global_md(r) for r in global_rows],
        ),
        encoding="utf-8",
    )

    (out_dir / "by_map_group.md").write_text(
        md_table(
            [
                "model_name",
                "source_run",
                "map_group",
                "episodes",
                "CSR",
                "ISR",
                "SR",
                "SoC_mean",
                "SoC_succ",
                "SoC_cm",
                "n_cm",
                "runtime_mean_s",
            ],
            [
                (
                    r["model_name"],
                    r["source_run"],
                    r["map_group"],
                    r["episodes"],
                    fmt_md_float(r["CSR_mean"], 4),
                    fmt_md_float(r["ISR_mean"], 4),
                    fmt_md_float(r["SR_mean"], 4),
                    fmt_md_float(r["SoC_mean"], 2),
                    fmt_md_float(r["SoC_success_only_mean"], 2),
                    fmt_md_float(r["SoC_on_common_success_with_baseline"], 2),
                    str(r.get("SoC_common_pair_episodes_n", "")),
                    fmt_md_float(r["runtime_mean"], 3),
                )
                for r in by_group_rows
            ],
        ),
        encoding="utf-8",
    )

    (out_dir / "by_map_group_agents.md").write_text(
        md_table(
            [
                "model",
                "run",
                "map_group",
                "agents",
                "n",
                "CSR",
                "CSR_CI",
                "ISR",
                "ISR_CI",
                "SR",
                "SR_CI",
                "SoC_mean",
                "SoC_succ",
                "SoC_cm",
                "n_cm",
                "rt_s",
            ],
            [
                (
                    r["model_name"],
                    r["source_run"],
                    r["map_group"],
                    r["num_agents"],
                    r["episodes"],
                    fmt_md_float(r["CSR_mean"], 4),
                    f"[{fmt_md_float(r['CSR_ci95_low'], 4)},{fmt_md_float(r['CSR_ci95_high'], 4)}]"
                    if r["episodes"] > 1
                    else "",
                    fmt_md_float(r["ISR_mean"], 4),
                    f"[{fmt_md_float(r['ISR_ci95_low'], 4)},{fmt_md_float(r['ISR_ci95_high'], 4)}]"
                    if r["episodes"] > 1
                    else "",
                    fmt_md_float(r["SR_mean"], 4),
                    f"[{fmt_md_float(r['SR_ci95_low'], 4)},{fmt_md_float(r['SR_ci95_high'], 4)}]"
                    if r["episodes"] > 1
                    else "",
                    fmt_md_float(r["SoC_mean"], 2),
                    fmt_md_float(r["SoC_success_only_mean"], 2),
                    fmt_md_float(r["SoC_on_common_success_with_baseline"], 2),
                    str(r.get("SoC_common_pair_episodes_n", "")),
                    fmt_md_float(r["runtime_mean"], 3),
                )
                for r in by_ga_rows
            ],
        ),
        encoding="utf-8",
    )

    (out_dir / "relative_to_baseline_global.md").write_text(
        md_table(
            [
                "model_name",
                "source_run",
                "ΔCSR",
                "ΔISR",
                "ΔSR",
                "ΔSoC_succ",
                "ΔSoC_cm",
                "n_cm",
                "SoC_ratio",
                "rt_ratio",
                "speedup",
                "ok_1%",
                "ok_3%",
                "ok_5%",
            ],
            [
                (
                    r["model_name"],
                    r["source_run"],
                    fmt_md_float(r["CSR_delta_vs_baseline"], 4),
                    fmt_md_float(r["ISR_delta_vs_baseline"], 4),
                    fmt_md_float(r["SR_delta_vs_baseline"], 4),
                    fmt_md_float(r["SoC_success_only_delta_vs_baseline"], 4),
                    fmt_md_float(r["SoC_common_delta_pairwise"], 4),
                    str(r.get("SoC_common_pair_episodes_n", "")),
                    fmt_md_float(r["SoC_success_only_ratio_vs_baseline"], 4),
                    fmt_md_float(r["runtime_ratio_vs_baseline"], 4),
                    fmt_md_speedup(r["speedup_vs_baseline"]),
                    str(r["quality_pass_epsilon_1pct"]),
                    str(r["quality_pass_epsilon_3pct"]),
                    str(r["quality_pass_epsilon_5pct"]),
                )
                for r in rel_global
            ],
        ),
        encoding="utf-8",
    )

    (out_dir / "relative_to_baseline_by_group.md").write_text(
        md_table(
            [
                "model_name",
                "source_run",
                "map_group",
                "ΔCSR",
                "ΔISR",
                "ΔSR",
                "ΔSoC_succ",
                "ΔSoC_cm",
                "n_cm",
                "rt_ratio",
                "speedup",
            ],
            [
                (
                    r["model_name"],
                    r["source_run"],
                    r["map_group"],
                    fmt_md_float(r["CSR_delta_vs_baseline"], 4),
                    fmt_md_float(r["ISR_delta_vs_baseline"], 4),
                    fmt_md_float(r["SR_delta_vs_baseline"], 4),
                    fmt_md_float(r["SoC_success_only_delta_vs_baseline"], 4),
                    fmt_md_float(r["SoC_common_delta_pairwise"], 4),
                    str(r.get("SoC_common_pair_episodes_n", "")),
                    fmt_md_float(r["runtime_ratio_vs_baseline"], 4),
                    fmt_md_speedup(r["speedup_vs_baseline"]),
                )
                for r in rel_by_g
            ],
        ),
        encoding="utf-8",
    )

    template_header = (
        "model_name,method,params,precision,source_run,episodes,CSR_mean,ISR_mean,SR_mean,"
        "SoC_success_only_mean,SoC_on_common_success_with_baseline,runtime_mean,speedup_vs_2M,CSR_delta_vs_2M,notes\n"
    )
    (out_dir / "comparison_template.csv").write_text(template_header, encoding="utf-8")

    readme_lines = [
        "# Comparison tables",
        "",
        "## Inputs",
        "",
        "Run directories:",
        "",
        *[f"- `{d}`" for d in run_dirs],
        "",
        f"- Baseline model for deltas: **`{args.baseline_model}`** (primary lookup in first run: `{first_label}`).",
        f"- Used **{'raw_results.jsonl' if used_raw else 'summary CSV fallback'}** where available.",
        "",
        "## Tables",
        "",
        "| File | Meaning |",
        "| --- | --- |",
        "| `global_by_model.*` | Metrics pooled over all map groups / agent counts per `(model_name, source_run)`. |",
        "| `by_map_group.*` | Same metrics grouped by `map_group`. |",
        "| `by_map_group_agents.*` | Grouped by `map_group` and `num_agents`; Wilson 95% CI for CSR/ISR/SR when episodes > 1. |",
        "| `relative_to_baseline_global.*` | Deltas vs baseline **global** aggregates; `speedup = baseline_runtime / model_runtime` (>1 if faster). |",
        "| `relative_to_baseline_by_group.*` | Same comparison **within each map_group**. |",
        "| `comparison_template.csv` | Empty template for filling future pipelines manually or by spreadsheet. |",
        "",
        "## Interpretation",
        "",
        "- **CSR / ISR / SR**: higher is better. **Runtime** (mean end-to-end episode time): lower is better.",
        "- **speedup_vs_baseline**: values **> 1** mean the model is faster than baseline (lower runtime).",
        "- **`SoC_mean`**: default = mean SoC over **all** episodes with valid SoC (includes failures), same spirit as "
        "`summary_by_model_map_agents.csv` aggregated `SoC_mean`.",
        "- **`SoC_success_only_mean`**: mean SoC over CSR-successful episodes only (success pool shifts with CSR — interpret carefully vs other models).",
        "- **`SoC_on_common_success_with_baseline`** (markdown `SoC_cm`): mean SoC for this model on episodes where **both** "
        f"this model and **`{args.baseline_model}`** (from the first `--runs` dir) CSR-succeed; baseline SoC on that same set is "
        "`SoC_baseline_on_same_common_episodes` (`SoC_bl_cm` in `global_by_model.md`). Use this for fair compression / distillation comparison.",
        "- **`--soc-mean-success-only`**: if set, `SoC_mean` is restricted to successful episodes (often ≈ `SoC_success_only_mean`).",
        "- **SoC** ratio vs baseline (success-only column): lower SoC is better; ratio is model/baseline on success-only means.",
        "- **Quality pass**: CSR not worse than baseline by more than ε percentage points: "
        "`CSR_model ≥ CSR_baseline − ε/100`; ε ∈ {1,3,5}.",
        "",
        "## Runtime note",
        "",
        "`runtime_mean` is **POGEMA end-to-end episode time** (environment stepping + policy), not isolated neural net forward latency.",
        "",
        "## Adding new methods",
        "",
        "Save new runs with the same JSONL schema as `extras/run_author_baselines.py` (per-episode CSR/ISR/SR/SoC/runtime), then re-run this script with multiple `--runs` directories.",
        "",
    ]
    if args.drop_puzzles:
        readme_lines.insert(4, "*Filtered: excluded `05-puzzles`.*")
    if args.only_groups:
        readme_lines.insert(4, f"*Filtered: only groups {list(args.only_groups)}.*")
    if warnings:
        readme_lines.extend(["", "## Warnings", ""])
        readme_lines.extend(f"- {w}" for w in warnings)

    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    if args.verbose:
        print(f"Wrote tables under {out_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
