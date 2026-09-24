"""
McNemar power analysis for hallucination-rate changes (300 × 3 categories).

McNemar power analysis for main experiments (n=300 per task).
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from scipy.stats import norm

from scripts.config import load_path_settings

logger = logging.getLogger(__name__)

ALPHA = 0.05
TARGET_POWER = 0.80
SAMPLES_PER_CATEGORY = 300
N_CATEGORIES = 3
TOTAL_SAMPLES = SAMPLES_PER_CATEGORY * N_CATEGORIES

CATEGORIES = {
    "para": "para_sampled_300.jsonl",
    "sum": "sum_sampled_300.jsonl",
    "d2t": "d2t_sampled_300.jsonl",
}


def norm_cdf(x: float) -> float:
    return norm.cdf(x)


def norm_ppf(p: float) -> float:
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    return norm.ppf(p)


def mcnemar_power(
    n: int,
    p_initial: float,
    p_final: float,
    alpha: float = ALPHA,
) -> float:
    if n < 1:
        return alpha
    delta = p_initial - p_final
    if delta <= 0:
        return alpha
    c_exp = n * delta
    b_exp = 0.0
    n_disc = c_exp + b_exp
    if n_disc < 0.5:
        return alpha
    z_crit = norm_ppf(1 - alpha)
    z_effect = (abs(c_exp - b_exp) - 1) / math.sqrt(n_disc)
    return 1 - norm_cdf(z_crit - z_effect)


def mcnemar_mde_rate(
    n: int,
    alpha: float = ALPHA,
    power: float = TARGET_POWER,
) -> float:
    if n < 1:
        return float("inf")
    z_needed = norm_ppf(1 - alpha) + norm_ppf(power)
    lo, hi = 1.0, float(n)
    for _ in range(60):
        mid = (lo + hi) / 2
        if (mid - 1) / math.sqrt(mid) < z_needed:
            lo = mid
        else:
            hi = mid
    return hi / n


def median(values: List[float]) -> float:
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def extract_task_id(entry_id: str) -> str:
    match = re.match(r"(task\d+)", entry_id)
    return match.group(1) if match else entry_id


def load_dataset_task_stats(data_dir: Path) -> Dict[str, Any]:
    stats_by_cat: Dict[str, Any] = {}
    for cat, fname in CATEGORIES.items():
        path = data_dir / fname
        tasks: List[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                tasks.append(extract_task_id(json.loads(line)["id"]))
        counter = Counter(tasks)
        counts = sorted(counter.values())
        stats_by_cat[cat] = {
            "n_samples": len(tasks),
            "n_unique_tasks": len(counter),
            "items_per_task": {
                "min": min(counts),
                "max": max(counts),
                "median": counts[len(counts) // 2],
            },
            "task_counts": dict(counter),
        }
    stats_by_cat["total"] = {
        "n_samples": sum(s["n_samples"] for s in stats_by_cat.values()),
        "n_unique_tasks": sum(s["n_unique_tasks"] for s in stats_by_cat.values()),
    }
    return stats_by_cat


def load_all_hallucination_effects(result_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    mde_pp = mcnemar_mde_rate(SAMPLES_PER_CATEGORY) * 100

    for path in sorted(result_dir.rglob("summary_new_batch.json")):
        rel = path.relative_to(result_dir)
        if len(rel.parts) >= 2 and rel.parts[0] in CATEGORIES:
            category, model_dir = rel.parts[0], rel.parts[1]
        else:
            category = rel.parts[0] if rel.parts and rel.parts[0] in CATEGORIES else "all"
            model_dir = rel.parent.name if rel.parent != Path(".") else "default"
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for method, md in data.items():
            if not isinstance(md, dict):
                continue
            if "avg_initial_hallucination_rate" not in md:
                continue
            p_ini = md["avg_initial_hallucination_rate"] / 100
            p_fin = md["avg_final_hallucination_rate"] / 100
            drop_pp = (p_ini - p_fin) * 100
            power = mcnemar_power(SAMPLES_PER_CATEGORY, p_ini, p_fin)
            records.append(
                {
                    "category": category,
                    "model": model_dir,
                    "method": method,
                    "drop_pp": drop_pp,
                    "power": power,
                    "detectable": drop_pp >= mde_pp,
                    "adequate_power": power >= TARGET_POWER,
                }
            )
    return records


def run_power_analysis(
    data_dir: Path | None = None,
    result_dir: Path | None = None,
) -> Dict[str, Any]:
    paths = load_path_settings()
    data_dir = data_dir or Path(paths.data_dir)
    result_dir = result_dir or Path(paths.result_dir)

    task_stats = load_dataset_task_stats(data_dir)
    effects = load_all_hallucination_effects(result_dir)

    mde_300 = mcnemar_mde_rate(SAMPLES_PER_CATEGORY) * 100
    mde_900 = mcnemar_mde_rate(TOTAL_SAMPLES) * 100

    scenario_powers = {
        delta: mcnemar_power(SAMPLES_PER_CATEGORY, 0.17, 0.17 - delta / 100)
        for delta in [2, 3, 4, 5, 7, 10]
    }

    task_ns = [
        count
        for cat, info in task_stats.items()
        if cat != "total"
        for count in info["task_counts"].values()
    ]
    worst_task_mde = max(mcnemar_mde_rate(n) * 100 for n in task_ns)

    drops = [r["drop_pp"] for r in effects]
    ge3 = [r for r in effects if r["drop_pp"] >= 3.0]
    ge3_powered = [r for r in ge3 if r["adequate_power"]]
    adequate = [r for r in effects if r["adequate_power"]]

    report = {
        "sample_design": {
            "per_category_n": SAMPLES_PER_CATEGORY,
            "pooled_n": TOTAL_SAMPLES,
            "n_categories": N_CATEGORIES,
            "n_unique_subtasks": task_stats["total"]["n_unique_tasks"],
            "subtask_n_min": min(task_ns),
            "subtask_n_median": int(median(task_ns)),
        },
        "mde_pp_80_power": {
            "n_300": mde_300,
            "n_900": mde_900,
            "worst_subtask_illustrative": worst_task_mde,
        },
        "scenario_power_baseline_17pct": {
            str(delta): pow_ for delta, pow_ in scenario_powers.items()
        },
        "observed_effects": {
            "n_conditions": len(effects),
            "median_drop_pp": median(drops) if drops else None,
            "mean_drop_pp": statistics.mean(drops) if drops else None,
            "max_drop_pp": max(drops) if drops else None,
            "n_detectable_at_mde": sum(r["detectable"] for r in effects),
            "n_adequate_power": len(adequate),
            "n_ge3pp_with_80_power": len(ge3_powered),
            "n_ge3pp_total": len(ge3),
            "top5": sorted(effects, key=lambda r: r["drop_pp"], reverse=True)[:5],
        },
    }

    logger.info("Power analysis complete: %d effect conditions loaded", len(effects))
    return report


def main() -> Dict[str, Any]:
    return run_power_analysis()


if __name__ == "__main__":
    from scripts.logging_config import configure_logging

    configure_logging()
    main()
