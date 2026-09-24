"""Task router: dispatch to eval.para | eval.sum | eval.d2t."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from . import d2t, para, sum

_TASKS = {
    "para": para,
    "sum": sum,
    "d2t": d2t,
}


def get_task_config(task: str) -> Dict[str, str]:
    if task not in _TASKS:
        raise ValueError(f"Unknown task: {task}. Expected one of {list(_TASKS)}")
    return _TASKS[task].JSONL_KEYS


def evaluate_jsonl(task: str, jsonl_path: str) -> Dict[str, Any]:
    return _TASKS[task].evaluate_jsonl(jsonl_path)


def evaluate_jsonl_pair(task: str, jsonl_path: str) -> Tuple[float, float]:
    return _TASKS[task].evaluate_jsonl_pair(jsonl_path)
