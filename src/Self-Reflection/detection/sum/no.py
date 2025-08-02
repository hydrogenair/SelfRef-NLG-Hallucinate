
import json
import re
import os
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

class ContentSafetyError(Exception):
    """Content safety detection exception"""

def create_client():
    return OpenAI(
        base_url = "",
        api_key = ""
    )

def call_api(client: OpenAI, prompt: str) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except APIError as e:
        if e.code == 400 and getattr(e, 'type', '') == 'data_inspection_failed':
            raise ContentSafetyError("Content safety detection failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def generate_initial_summary(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_api(client, prompt)
        match = re.search(r'Summary:\s*(.+)$', response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()
    except ContentSafetyError:
        print(f"Content safety blocked: first 20 characters of article {question[:20]}...")
        return None



def regenerate_summary(client: OpenAI, article: str, initial_summary: str) -> Optional[str]:
    try:
        prompt = f"""Generate a revised final summary based on the original request and initial summary.
        The initial answer may contain inaccuracies or hallucinations.
        Request: {article}
        Initial Summary (May contain errors):{initial_summary}
        Generate the revised final summary:"""
        
        
        response = call_api(client, prompt)
        return response.strip()
    except ContentSafetyError:
        return None
    
def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"][0]
        
        # Generate initial answer
        initial = generate_initial_summary(client, question)
        if not initial:
            return None
            
        # Generate final answer directly
        final_answer = regenerate_summary(client, question, initial)
        
        return {
            "input": input_text,
            "question": question,
            "initial_summary": initial,
            "final_summary": final_answer if final_answer else "[Blocked]",
            "reference_summary": reference
        }
    except ContentSafetyError:
        return {
            "input": input_text,
            "question": question,
            "initial_summary": "[Blocked]",
            "final_summary": "[Blocked]",
            "reference_summary": reference
        }
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None

def main(input_file: str, output_file: str):
    client = create_client()
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                if processed := process_entry(client, entry):
                    outfile.write(json.dumps(processed, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
            except Exception as e:
                print(f"Main process error: {str(e)}")


if __name__ == "__main__":
    INPUT_JSONL = "data/processed/detector/sum_sampled_50.jsonl"
    OUTPUT_JSONL = "results/Self-Reflection/sum/llama/sum_llama_no_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)