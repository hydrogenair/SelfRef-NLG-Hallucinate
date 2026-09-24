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
        return response.strip() if response else None
    except ContentSafetyError:
        logger.warning(f"Content blocked: {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str) -> Dict:
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. You will be given two inputs: Structured Data and Generated Text, a natural language description of the data.
You are asked to evaluate the Generated Text by looking at the Structured Data.

Then, you need to answer the question: is the Generated Text a hallucination or not?
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.
Your mission is to Score the generated text quality from 1-10 based on faithfulness to the original text.
1-3: Severe hallucination (key facts altered/missing)
4-6: Partial hallucination (minor inaccuracies)
7-10: Factually consistent (preserves core meaning)

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
score: 2

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
score: 9

Respond STRICTLY in the format:
Score: [1-10]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}"""
        
        response = _call(client, prompt)
        score_match = re.search(r'Score:\s*(\d+)', response)
        return {"score": int(score_match.group(1)) if score_match else 1}
    
    except Exception as e:
        logger.warning(f"Detection error: {str(e)}")
        return {"score": 1}

def regenerate_answer(client: OpenAI, data: str, wrong_text: str, score: int) -> Optional[str]:
    try:
        prompt = f"""
        Request: {data}
        Invalid text: {wrong_text}
        Initial generated text scored {score}/10 (threshold {HALLUCINATION_THRESHOLD}), which means the initial generated text contains inaccuracies or hallucinations.
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
        input_text = entry["input"]
        reference = entry["answer"][0]

        initial = generate_initial(client, question)
        if not initial:
            return None
    
        detection = detect_hallucination(client, question, initial)
        needs_correction = detection["score"] < HALLUCINATION_THRESHOLD
        
        final_answer = initial
        if needs_correction:
            if corrected := regenerate_answer(client, question, initial, detection["score"]):
                final_answer = corrected
        
        return {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "score": detection["score"],
            "final_answer": final_answer,
            "reference_answer": reference
        }
    except ContentSafetyError:
        return {
            "input": input_text,
            "question": question,
            "initial_answer": "[Blocked]",
            "hallucination": "yes",
            "final_answer": "[Blocked]",
            "reference_answer": reference
        }
    except Exception as e:
        logger.warning(f"Processing error: {str(e)}")
        return None


def main(input_file: str, output_file: str) -> None:
    client = get_client()
    run_jsonl_main(client, input_file, output_file, process_entry)
