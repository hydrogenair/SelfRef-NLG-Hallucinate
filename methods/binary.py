"""Data-to-text reflection method (publication release)."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from openai import OpenAI

from scripts.jsonl_pipeline import run_jsonl_main
from scripts.llm_client import ContentSafetyError, chat_completion, get_client

logger = logging.getLogger(__name__)

HALLUCINATION_THRESHOLD = 5


def _call(client: OpenAI, prompt: str) -> Optional[str]:
    return chat_completion(client, prompt)


def generate_initial(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = _call(client, prompt)
        return response.strip()
    except ContentSafetyError:
        logger.warning("Content safety blocked (question preview): %s...", question[:20])
        return None

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str) -> Optional[bool]:
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. You will be given two inputs: Structured Data and Generated Text, a natural language description of the data.
You are asked to evaluate the Generated Text by looking at the Structured Data.

Then, you need to answer the question: is the Generated Text a hallucination or not?
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Hallucination: Yes

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: No

Final judgment responds STRICTLY in this format: 
Hallucination: [Yes/No]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""
        response = _call(client, prompt).strip()
        return response.lower().startswith("hallucination: yes")
    except ContentSafetyError:
        return None
    except Exception as e:
        logger.warning(f"Detection Error: {str(e)}")
        return True



def regenerate_answer(client: OpenAI, data: str, wrong_text: str) -> Optional[str]:
    try:
        prompt = f"""
        Request: {data}
        Invalid text: {wrong_text}
        This generated text contains inaccuracies. 
        Generate a corrected version using ONLY information from the Structured Data. 
        Respond STRICTLY in this format: 
        Corrected text: [your text here]
"""
        
        response = _call(client, prompt)
        if match := re.search(r'Corrected text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        return None
    except ContentSafetyError:
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_data = entry["input"]
        reference = entry["answer"][0]
        
        # Generate initial 
        initial = generate_initial(client, question)
        if initial is None:
            return None
            
        # Detect hallucinations
        has_hallu = detect_hallucination(client, question, initial)
        if has_hallu is None:
            return None
            
        # Build result record
        result = {
            "input": input_data,
            "question": question,
            "initial_answer": initial,
            "hallucination": "yes" if has_hallu else "no",
            "reference_answer": reference
        }

        if has_hallu:
            if corrected := regenerate_answer(client, question, initial):
                result["final_answer"] = corrected
            else:
                return None
        else:
            result["final_answer"] = initial

        return result
        
    except ContentSafetyError:
        return None


def main(input_file: str, output_file: str) -> None:
    client = get_client()
    run_jsonl_main(client, input_file, output_file, process_entry)
