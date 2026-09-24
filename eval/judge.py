"""CLI: compute initial/final hallucination rates on a result JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import os

from eval.hallucination import evaluate_jsonl

_TASKS = ("para", "sum", "d2t")
from scripts.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=_TASKS, required=True)
    parser.add_argument("--input-jsonl", default=os.environ.get("EVAL_INPUT_JSONL"))
    parser.add_argument("--output-json", default=os.environ.get("EVAL_OUTPUT_JSON"))
    args = parser.parse_args()
    if not args.input_jsonl:
        raise RuntimeError("Set --input-jsonl or EVAL_INPUT_JSONL")

    report = evaluate_jsonl(args.task, args.input_jsonl)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    logger.info("Evaluation report: %s", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
