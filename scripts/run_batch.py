"""
Batch runner: apply all reflection methods on a task dataset and evaluate hallucination rates.

Environment variables:
  TASK              para | sum | d2t
  INPUT_JSONL       defaults to data/{task}_sampled_300.jsonl under DATA_DIR
  RESULT_DIR        output directory
  DETECT_METHODS_DIR  directory containing method *.py modules (default: methods)
  RUNS              repetitions per method (default: 1)
  OUTPUT_PREFIX     filename prefix (default: task name)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from eval.hallucination import evaluate_jsonl_pair, get_task_config
from scripts.config import default_dataset_path, load_path_settings
from scripts.logging_config import configure_logging

logger = logging.getLogger(__name__)


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_detect_scripts(methods_dir: Path) -> List[str]:
    return sorted(
        f.name
        for f in methods_dir.iterdir()
        if f.suffix == ".py" and not f.name.startswith("_")
    )


def run_batch(
    *,
    task: str,
    input_jsonl: str,
    result_dir: str,
    methods_dir: str,
    runs: int,
    output_prefix: str | None = None,
) -> Dict[str, Any]:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    methods_path = Path(methods_dir)
    prefix = output_prefix or task
    get_task_config(task)

    summary: Dict[str, Any] = {}
    summary_file = result_path / "summary_new_batch.json"

    for script_name in list_detect_scripts(methods_path):
        stem = Path(script_name).stem
        detect_module = load_module_from_path(
            f"detect_{stem.replace('-', '_')}",
            str(methods_path / script_name),
        )
        if not hasattr(detect_module, "main"):
            logger.warning("Skip %s: missing main(input_file, output_file)", script_name)
            continue

        logger.info("Running method: %s", script_name)
        per_runs: List[Dict[str, float]] = []

        for run_idx in range(1, runs + 1):
            output_file = result_path / f"{prefix}_{stem}_300_run{run_idx}.jsonl"
            detect_module.main(input_jsonl, str(output_file))

            init_h, final_h = evaluate_jsonl_pair(task, str(output_file))

            per_runs.append(
                {
                    "run_idx": run_idx,
                    "initial_hallucination_rate": init_h,
                    "final_hallucination_rate": final_h,
                }
            )

            summary[script_name] = {
                "runs": per_runs,
                "avg_initial_hallucination_rate": sum(r["initial_hallucination_rate"] for r in per_runs)
                / len(per_runs),
                "avg_final_hallucination_rate": sum(r["final_hallucination_rate"] for r in per_runs)
                / len(per_runs),
            }
            with open(summary_file, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)

            logger.info(
                "Run %d complete: init_h=%.2f%% final_h=%.2f%%",
                run_idx,
                init_h,
                final_h,
            )

    return summary


def main() -> Dict[str, Any]:
    configure_logging()
    paths = load_path_settings()

    parser = argparse.ArgumentParser(description="Batch reflection experiment runner")
    parser.add_argument("--task", default=os.environ.get("TASK", "d2t"))
    parser.add_argument(
        "--input-jsonl",
        default=os.environ.get("INPUT_JSONL"),
        help="Input JSONL; default data/{task}_sampled_300.jsonl",
    )
    parser.add_argument("--result-dir", default=os.environ.get("RESULT_DIR", paths.result_dir))
    parser.add_argument(
        "--methods-dir",
        default=os.environ.get("DETECT_METHODS_DIR", paths.detect_methods_dir),
    )
    parser.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "1")))
    parser.add_argument("--output-prefix", default=os.environ.get("OUTPUT_PREFIX"))
    args = parser.parse_args()

    input_jsonl = args.input_jsonl or default_dataset_path(args.task, n=300)

    return run_batch(
        task=args.task,
        input_jsonl=input_jsonl,
        result_dir=args.result_dir,
        methods_dir=args.methods_dir,
        runs=args.runs,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
