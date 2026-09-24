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
        return  response.strip()
    except ContentSafetyError:
        logger.warning("Content safety blocked (question preview): %s...", question[:20])
        return None

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str) -> Dict:
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. You will be given two inputs: Structured Data and Generated Text, a natural language description of the data.
You are asked to evaluate the Generated Text by looking at the Structured Data.

Then, you need to answer the question: is the Generated Text a hallucination or not? If it's a hallucination, you need to generate precise feedback for correction.
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Feedback: the original data states '"temperature": 23.5' but the text says '25°C'. Correct the age to match the source.


Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Feedback:None



Respond STRICTLY in this format:
Feedback: [Specific feedback content or None]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""

        response = _call(client, prompt).strip()
        return response.lower().startswith("Feedback") 
    except ContentSafetyError:
        return None
    
    
def regenerate(client: OpenAI, data: str, wrong_text: str, feedback: str) -> Optional[str]:
    try:
        prompt = f"""
Request: {data}
Invalid text: {wrong_text}
This generated text contains inaccuracies. 
Critical Feedback: {feedback}
Generate a corrected version using ONLY information from the Structured Data. 
Respond STRICTLY in this format: 
Corrected text: [your text here] 
"""

        response = _call(client, prompt)
        return response.strip()
    except ContentSafetyError:
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"]
        
        # Generate initial
        initial = generate_initial(client, question)
        if initial is None:
            return None
            
        # Detect hallucinations
        feedback = detect_hallucination(client, question, initial)
        if feedback is None:
            return 'None'
            
        # Build result record
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "feedback": feedback,
            "reference_answer": reference

        }

        if feedback:
            if corrected := regenerate(client, question, initial):
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
