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

Then, you need to answer the question: is the Generated Text a hallucination or not?If it's a hallucination, you need to find the location where the hallucination appears.
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Hallucination: Yes
Location:
Hallucinated Content: 25°C
Evidence: "temperature": 23.5


Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: No
Location: None

Final judgment responds STRICTLY in this format:
Hallucination: [Yes/No]
Location:
Hallucinated Content: "[exact text]"
Evidence: "[relevant text]" 
... (only if Hallucination is Yes)

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}"""

        response = _call(client, prompt).strip()
        
        has_hallucination = None
        locations = []
    
        lines = [line.strip() for line in response.split('\n') if line.strip()]
    
        if lines and lines[0].startswith("Hallucination:"):
            hallucination_status = lines[0].split(":", 1)[1].strip().lower()
            has_hallucination = hallucination_status == "yes"
        
            if has_hallucination and len(lines) > 1 and lines[1].startswith("Location:"):
                i = 2
                while i < len(lines):
                    if lines[i].startswith("Hallucinated Content:"):
                        claim = lines[i].split(":", 1)[1].strip().strip('"')
                        i += 1
                        if i < len(lines) and lines[i].startswith("Evidence:"):
                            ref = lines[i].split(":", 1)[1].strip().strip('"')
                            locations.append({"claim": claim, "reference": ref})
                        i += 1
                    else:
                        i += 1
                    
        return (has_hallucination, locations if has_hallucination else None)
    
    except ContentSafetyError:
        return (None, None)
    except Exception as e:
        logger.warning(f"Hallucination detection failed: {str(e)}")
        return (None, None)
    
    
def regenerate(client: OpenAI, data: str, wrong_text: str, positions:str) -> Optional[str]:
    try:
        errors_str = "\n".join(
        f"Discrepancy found:\n"
        f"Hallucinated Content: {loc['claim']}\n"
        f"Evidence: {loc['reference'] or 'No supporting evidence'}"
        for loc in positions
    )
        prompt = f"""
        Request: {data}
        Invalid text: {wrong_text}
        This text contains inaccuracies. The following discrepancies were found between the text and the original data:
{errors_str}
Generate a corrected version using ONLY information from the Structured Data. 
Respond STRICTLY in this format: 
Corrected text: [your text here] 
"""
        
        response = _call(client, prompt)
        if match := re.search(r'Corrected text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        else :
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
            "question": question,
            "initial_answer": initial,
            "hallucination": "yes" if has_hallu else "no",
            "reference_answer": reference,
            "position":positions
        }

        if has_hallu:
            if corrected := regenerate(client, question, initial, positions):
                result["final_answer"] = corrected
                result["hallucination_positions"] = positions
            else:
                return None  # Skip when content safety blocks regeneration
        else:
            result["final_answer"] = initial

        return result
        
    except ContentSafetyError:
        return None


def main(input_file: str, output_file: str) -> None:
    client = get_client()
    run_jsonl_main(client, input_file, output_file, process_entry)
