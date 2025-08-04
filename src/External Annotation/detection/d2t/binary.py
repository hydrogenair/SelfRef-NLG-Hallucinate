#d2t binary 300
import json
import re
import os
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

class ContentSafetyError(Exception):
    """Content safety detection exception"""

def create_qwen_client():
    return OpenAI(
        api_key="",
        base_url="",
    )

def create_deepseek_client():
    return OpenAI(
        api_key="",
        base_url="",
    )

def call_qwen_api(client: OpenAI, prompt: str) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
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
            raise ContentSafetyError("Content safety detection failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def generate_initial(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_qwen_api(client, prompt)
        return response.strip()
    except ContentSafetyError:
        print(f"Content safety blocked: first 20 characters of article {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str) -> Dict:
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
        response = call_deepseek_api(client, prompt).strip()
        return response.lower().startswith("hallucination: yes")
    except ContentSafetyError:
        return None
    except Exception as e:
        print(f"Detection Error: {str(e)}")
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
        
        response = call_qwen_api(client, prompt)
        if match := re.search(r'Corrected text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        return response.strip()  # Fallback processing
    except ContentSafetyError:
        return None

def process_entry(qwen_client: OpenAI, deepseek_client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_data = entry["input"]
        reference = entry["answer"][0]
        
        # Generate initial 
        initial = generate_initial(qwen_client, question)
        if initial is None:
            return None
            
        # Detect hallucinations
        has_hallu = detect_hallucination(deepseek_client, question, initial)
        
            
        # Modified output logic
        result = {
            "input": input_data,
            "question": question,
            "initial_answer": initial,
            "hallucination": "yes" if has_hallu else "no",
            "reference_answer": reference
        }

        if has_hallu:
            if corrected := regenerate_answer(qwen_client, question, initial):
                result["final_answer"] = corrected
            else:
                result["final_answer"] = initial  # If regeneration fails, use original answer
        else:
            result["final_answer"] = initial

        return result
        
    except ContentSafetyError:
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
                print(f"JSON parsing failed: {line}")
            except Exception as e:
                print(f"Main process exception: {str(e)}")

if __name__ == "__main__":
    INPUT_JSONL = "data/processed/detector/d2t_sampled_50.jsonl"
    OUTPUT_JSONL = "results/External Annotation/d2t/d2t_deepseek72b_binary_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)