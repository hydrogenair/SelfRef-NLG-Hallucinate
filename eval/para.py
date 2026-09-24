"""Paraphrase hallucination evaluation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from openai import OpenAI

from eval.util import (
    rates_to_pair,
    run_llm_judge,
    scan_jsonl_hallucination_rates,
)

TASK = "para"

JSONL_KEYS = {
    "input_key": "input",
    "initial_key": "initial_answer",
    "final_key": "final_answer",
    "reference_key": "reference_answer",
}

PROMPT = """
The given task is to evaluate whether the paraphrased text contains hallucinated content by comparing it with the original text.

You are a linguist specializing in semantic analysis. Your task is to compare the paraphrased text with the original text, focusing on factual consistency and meaning preservation. The paraphrased text should accurately reflect the information in the original, but it may use different wording, grammatical structures, and natural language paraphrasing. Minor variations in wording, reasonable generalizations, and syntactic transformations that preserve the core meaning are acceptable and should NOT be considered hallucinations.

**Key considerations:**
- Accept grammatical transformations that are semantically equivalent (e.g., dative alternation: "He gave the man the book" ↔ "He gave the book to the man"; "He kicked the man the ball" ↔ "He kicked the ball to the man").
- If the original text contains ungrammatical or non-standard phrasing, interpret its intended meaning based on context and common usage. The paraphrase may correct the grammar while preserving that intended meaning.
- Avoid overly literal interpretation that would reject synonymous expressions or equivalent syntactic variants.
- Focus on whether the paraphrase introduces new facts, changes entities or their relationships, or contradicts explicit information in the original.

A paraphrased text is considered hallucinated if it:
- Contains information that directly contradicts the original text.
- Introduces specific details, numbers, or claims that are not present in the original text and cannot be logically inferred from it.
- Distorts the relationships between entities or events, leading to a different interpretation than the original (e.g., changing who performed an action, altering the target of an action, or introducing a new event not implied).

Acceptable variations include:
- Using synonyms or alternative phrasing (e.g., "several weeks" for "some three weeks").
- Combining related facts into a concise statement (e.g., mentioning multiple details together).
- Making reasonable inferences that are strongly implied by the context.
- Changing grammatical structure while preserving meaning (e.g., active ↔ passive voice, double-object construction ↔ prepositional dative, etc.).

Example 1 (Hallucination):
Original Text: "The European Union has 27 member states."
Paraphrase: "There are 25 countries in the EU bloc."
Hallucination: yes
Reason: Incorrect member count (25 vs 27 in Original Text and Reference Answer).

Example 2 (Not Hallucination):
Original Text: "Global temperatures rose 1.1°C since 1880."
Paraphrase: "Earth's temperature has increased by 1.1 degrees Celsius since the late 19th century."
Hallucination: no
Reason: Facts align with both Original Text and Reference Answer.

**Now evaluate the following:**
Original Text: {source}
Paraphrased Text: {candidate}

Is the paraphrased text a hallucination? Respond strictly in the format:
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
