"""Shared helpers for task-specific hallucination eval modules."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Tuple

from openai import OpenAI

from scripts.llm_client import chat_completion, get_client

logger = logging.getLogger(__name__)

DetectFn = Callable[[OpenAI, str, str, str], Dict[str, str]]


def create_client() -> OpenAI:
    return get_client()


def parse_yes_no(response: str) -> Dict[str, str]:
    judgment_match = re.search(r"Hallucination:\s*(yes|no)", response, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", response, re.IGNORECASE)
    return {
        "gpt_final": judgment_match.group(1).capitalize() if judgment_match else "No",
        "gpt_reason": reason_match.group(1).strip() if reason_match else "No reason provided",
    }


def run_llm_judge(client: OpenAI, prompt: str, task_label: str) -> Dict[str, str]:
    try:
        response = chat_completion(client, prompt)
        if not response:
            return {"gpt_final": "No", "gpt_reason": "Empty model response"}
        return parse_yes_no(response)
    except Exception as exc:
        logger.warning("Hallucination check failed (%s): %s", task_label, exc)
        return {"gpt_final": "No", "gpt_reason": f"Error in detection: {exc}"}


def scan_jsonl_hallucination_rates(
    jsonl_path: str,
    *,
    task: str,
    input_key: str,
    initial_key: str,
    final_key: str,
    reference_key: str,
    detect_fn: DetectFn,
    client: OpenAI | None = None,
) -> Dict[str, Any]:
    client = client or create_client()
    total_initial = 0
    total_final = 0
    total_questions = 0

    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                total_questions += 1
                source = entry.get(input_key, "")
                ref = entry.get(reference_key, "")

                initial = entry.get(initial_key, "")
                if initial and initial != "[Blocked]":
                    if detect_fn(client, source, initial, ref).get("gpt_final") == "Yes":
                        total_initial += 1

                final = entry.get(final_key, "")
                if final and final != "[Blocked]":
                    if detect_fn(client, source, final, ref).get("gpt_final") == "Yes":
                        total_final += 1
            except json.JSONDecodeError:
                logger.warning("Invalid JSON skipped in %s", jsonl_path)
            except Exception as exc:
                logger.exception("Evaluation error: %s", exc)

    if total_questions == 0:
        return {
            "task": task,
            "total_questions": 0,
            "initial_hallucination_rate": 0.0,
            "final_hallucination_rate": 0.0,
        }

    return {
        "task": task,
        "total_questions": total_questions,
        "initial_hallucination_count": total_initial,
        "final_hallucination_count": total_final,
        "initial_hallucination_rate": total_initial / total_questions * 100.0,
        "final_hallucination_rate": total_final / total_questions * 100.0,
    }


def rates_to_pair(rates: Dict[str, Any]) -> Tuple[float, float]:
    return rates["initial_hallucination_rate"], rates["final_hallucination_rate"]
