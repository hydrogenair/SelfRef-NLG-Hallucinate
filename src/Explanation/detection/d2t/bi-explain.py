#d2t llama explanation
#binary-d2t
import json
import re
import os
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

class ContentSafetyError(Exception):
    """Content safety inspection exception"""

def create_qwen_client():
    return OpenAI(
        base_url = "",
        api_key = ""
    )

def create_deepseek_client():
    return OpenAI(
        api_key="",
        base_url="",
    )

def call_qwen_api(client: OpenAI, prompt: str) -> Optional[str]:
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
            raise ContentSafetyError("Content safety check failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def call_deepseek_api(client: OpenAI, prompt: str) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except APIError as e:
        if e.code == 400 and getattr(e, 'type', '') == 'data_inspection_failed':
            raise ContentSafetyError("Content safety check failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def generate_initial(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_qwen_api(client, prompt)
        return response.strip() if response else None
    except ContentSafetyError:
        print(f"Content blocked: {question[:20]}...")
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
Reason: Incorrect member count (25 vs 23.5 in Original Data and Generated Text).

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: No
Reason: Facts align with both Original Text and Reference Answer.

Final judgment responds STRICTLY in this format: 
Hallucination: [Yes/No]
Reason: [Your explanation]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""
        response = call_deepseek_api(client, prompt)
        
        # Extract judgment and reason
        judgment_match = re.search(r'Hallucination:\s*(yes|no)', response, re.IGNORECASE)
        reason_match = re.search(r'Reason:\s*(.+)', response, re.IGNORECASE)
        
        return {
            "gpt_final": judgment_match.group(1).capitalize() if judgment_match else "No",
            "gpt_reason": reason_match.group(1).strip() if reason_match else "No reason provided"
        }
    except Exception as e:
            print(f"Detection failed - {str(e)}")
            return {"gpt_final": "No", "gpt_reason": f"Error in detection: {str(e)}"}

def regenerate(client: OpenAI, data: str, wrong_text: str, reason: str) -> Optional[str]:
    try:
        prompt = f"""
 Request: {data}
Invalid text: {wrong_text}
This generated text contains inaccuracies.
The identified issue was: {reason}
Generate a corrected version using ONLY information from the Structured Data. 
Respond STRICTLY in this format: 
Corrected text: [your text here] """
        
        response = call_qwen_api(client, prompt)
        if match := re.search(r'Corrected text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        return response
    except ContentSafetyError:
        return None

def process_entry(qwen_client: OpenAI, deepseek_client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"]
        
        initial = generate_initial(qwen_client, question)
        if not initial:
            return None
            
        hallucination_result = detect_hallucination(deepseek_client, question, initial)
        has_hallucination = hallucination_result["gpt_final"].lower() == "yes"
        
        final_answer = initial
        if has_hallucination:
            if corrected := regenerate(
                qwen_client, 
                question, 
                initial,
                hallucination_result["gpt_reason"]
            ):
                final_answer = corrected
            else:
                final_answer = "[Content Blocked]"
        
        return {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "hallucination": "yes" if has_hallucination else "no",
            "final_answer": final_answer,
            "reason": hallucination_result["gpt_reason"],
            "reference_answer": reference
        }
    except ContentSafetyError:
        return {
            "input": input_text,
            "question": question,
            "initial_answer": "[Blocked]",
            "hallucination": "yes",
            "final_answer": "[Blocked]",
            "reason": hallucination_result["gpt_reason"],
            "reference_answer": reference
        }
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None

def main(input_file: str, output_file: str):
    qwen_client = create_qwen_client()
    deepseek_client = create_deepseek_client()
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                if processed := process_entry(qwen_client, deepseek_client, entry):
                    outfile.write(json.dumps(processed, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
            except Exception as e:
                print(f"Main process error: {str(e)}")

if __name__ == "__main__":
    INPUT_JSONL = "data/processed/detector/d2t_sampled_50.jsonl"
    OUTPUT_JSONL = "results/Explanation/d2t/d2t_deepseekllama_bi-explain_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)