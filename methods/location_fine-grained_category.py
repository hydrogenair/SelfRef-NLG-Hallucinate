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

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str) -> Dict:
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. You will be given two inputs: Structured Data and Generated Text, a natural language description of the data.
You are asked to evaluate the Generated Text by looking at the Structured Data.

Then, you need to answer the question: is the Generated Text a hallucination or not?if it's a hallucination, you need to find the location where the hallucination appears and the type of the hallucination.
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.
Hallucinations are categorized as follows:
        1. Incorrectly named entity(NAME): This includes people, places, organizations, and days of the week.
        2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
        3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
        4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
        5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
        6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)    
If the text is fully consistent and all information is supported by the article, the hallucination type is None.
For each claim in the text, determine if it falls into any of the above categories. After evaluating all claims, provide the final judgment by listing all applicable hallucination types separated by commas if multiple. If there are none, state None.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Hallucination: NUMBER
Location:
Hallucinated Content: 25°C
Evidence: "temperature": 23.5


Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: None
Location: None


Final Judgement can only select one or more of the following answers: NAME, NUMBER, WORD, CONTEXT, OTHER, NOT CHECKABLE, None.
Responds STRICTLY in this format:
Hallucination: [Your Final Judgement]
Location:
Hallucinated Content: "[exact text]"
Evidence: "[relevant text]" 
... (only if Hallucination is Yes)

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""

        
        response = _call(client, prompt).strip()
        
        hallucination_types = []
        errors = []
        
        # Extract primary hallucination type(s)
        type_line = next((line for line in response.split('\n') if line.startswith("Hallucination:")), None)
        if type_line:
            raw_types = type_line.split(":", 1)[1].strip()
            if raw_types != "None":
                hallucination_types = list(set([t.strip() for t in raw_types.split(",")]))
        
        # Parse per-span hallucination details
        current_error = None
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("Hallucinated Content:"):
                current_error = {
                    "types": [],
                    "claim": line.split(":", 1)[1].strip().strip('"'),
                    "evidence": None
                }
            elif line.startswith("Hallucination:") and current_error:
                current_error["types"] = [t.strip() for t in line.split(":",1)[1].split(",")]
            elif line.startswith("Evidence:") and current_error:
                current_error["evidence"] = line.split(":",1)[1].strip().strip('"')
                errors.append(current_error)
                current_error = None
        
        # Final type label
        final_types = "None" if not hallucination_types else ",".join(sorted(hallucination_types))
        return (final_types, errors if errors else [])
    
    except Exception as e:
        logger.warning(f"Detection error: {str(e)}")
        return ("None", [])
    
    
def regenerate(client: OpenAI, data: str, wrong_text: str, errors:str) -> Optional[str]:
    try:
        errors_str = []
        for error in errors:
            detail = (
                f"Type: {', '.join(error['types'])}\n"
                f"Hallucinated Content: {error['claim']}\n"
                f"Evidence: {error['evidence'] or 'No supporting evidence'}"
            )
            errors_str.append(detail)
        prompt = f"""
        Request: {data}
        Invalid text: {wrong_text}
        Hallucinations are categorized as follows:
        1. Incorrectly named entity(NAME): This includes people, places, organizations, and days of the week.
        2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
        3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
        4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
        5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
        6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)    
        This text contains inaccuracies. The following discrepancies were found between the text and the original data:
{errors_str}
Generate a corrected version using ONLY information from the Structured Data. 
Respond STRICTLY in this format: 
Corrected text: [your text here] 
"""

        response = _call(client, prompt)
        if response is None:
            return None
        return response.strip()
    except ContentSafetyError:
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"]
        
        
        initial = generate_initial(client, question)
        if initial is None:
            return None
            
        # Detect hallucinations
        has_hallu, positions = detect_hallucination(client, question, initial)
        if has_hallu is None:
            return None
            
        result = {
            "input": input_text,
            "question": entry["question"],
            "initial_answer": initial,
            "reference_answer": reference,
            "hallucination_types": "None",
            "hallucination_details": None
        }
        
        if has_hallu not in ["None"] and len(positions) > 0:
            result.update({
                "hallucination_types": has_hallu,
                "hallucination_details": [
                    {
                        "type": error.get("types", []),
                        "claim": error.get("claim", ""),
                        "evidence": error.get("evidence") or "No supporting evidence"
                    } for error in positions
                ]
            })
        
        # Regenerate when hallucination spans are present
        if has_hallu not in ["None"] and len(positions) > 0:
            if corrected := regenerate(client, question, initial, positions):
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
