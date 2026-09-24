"""Data-to-text hallucination evaluation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from openai import OpenAI

from eval.util import (
    rates_to_pair,
    run_llm_judge,
    scan_jsonl_hallucination_rates,
)

TASK = "d2t"

JSONL_KEYS = {
    "input_key": "input",
    "initial_key": "initial_answer",
    "final_key": "final_answer",
    "reference_key": "reference_answer",
}

PROMPT = """
The given task is to evaluate whether the generated text contains hallucinated content by comparing it with the original structured data.

You are a data accuracy expert. Your task is to compare the generated text with the structured data, focusing on factual consistency. The generated text should accurately reflect the information in the structured data, but it may use different wording, reasonable generalizations, and natural language paraphrasing. Minor variations in wording, rounding, or slight generalizations that preserve the core meaning are acceptable and should NOT be considered hallucinations.

Special note for concept-set tasks:
The structured data is a set of concepts (e.g., "fly#island#lake") that need to be combined into a coherent sentence describing a day-to-day scene. This inherently requires adding reasonable contextual details—such as time, place, actors, actions, or descriptive elements—to form a natural and plausible scene. Such additions are **allowed** as long as they are consistent with the concepts, do not contradict them, and are commonly associated with the concepts in everyday situations.

These additions do not alter the core meaning of the concepts and are part of constructing a realistic scenario. They should **not** be considered hallucinations.

A generated text is considered **hallucinated** if it:

- Contains information that directly contradicts the structured data.
- Introduces specific details, numbers, entities, or claims that are **not** present in any part of the structured data and cannot be logically inferred from it, **and** these details are unnecessary for building a coherent scene or are unrelated to the concepts.
- Distorts the relationships between data points or entities, leading to a different interpretation than the structured data provides.

Acceptable variations include:

- Using synonyms or alternative phrasing (e.g., "high temperature" for "35 degrees Celsius").
- Combining related data points into a concise statement.
- Performing accurate calculations based on the data.
- Making **reasonable inferences** strongly implied by the context or common sense when constructing a scene from a set of concepts.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"

Now evaluate the following:

Structured Data: {source}
Generated Text: {candidate}

Is the generated text a hallucination? Respond strictly in the format:
Hallucination: [yes/no]
Reason: [Your explanation]"""


def detect_hallucination(
    client: OpenAI,
    source: str,
    candidate: str,
    reference: str = "",
) -> Dict[str, str]:
    del reference
    prompt = PROMPT.format(source=source, candidate=candidate)
    return run_llm_judge(client, prompt, TASK)


def detect_paraphrase_hallucination(
    client: OpenAI,
    source: str,
    candidate: str,
    reference: str,
) -> Dict[str, str]:
    return detect_hallucination(client, source, candidate, reference)


def evaluate_jsonl(jsonl_path: str, client: OpenAI | None = None) -> Dict[str, Any]:
    return scan_jsonl_hallucination_rates(
        jsonl_path,
        task=TASK,
        detect_fn=detect_hallucination,
        client=client,
        **JSONL_KEYS,
    )


def evaluate_jsonl_pair(jsonl_path: str) -> Tuple[float, float]:
    return rates_to_pair(evaluate_jsonl(jsonl_path))
