#para position 300 llama
'''{
  "hallucination_types": "yes",
  "hallucination_details": [
    {
      "Text claim": "25°C",
      "Original Data evidence": ""temperature": 23.5'"
    },
    {
      "Text claim": "at 3 PM",
      "Original Data evidence": "no supporting evidence"
    }
  ]
} '''
import re
import os
import json
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

HALLUCINATION_THRESHOLD = 5

class ContentSafetyError(Exception):
    """content safety detection failed"""

def create_client():
    return OpenAI(
        base_url = "",
        api_key = ""
    )

def call_api(client: OpenAI, prompt: str) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model="qwen2.5-plus-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except APIError as e:
        if e.code == 400 and getattr(e, 'type', '') == 'data_inspection_failed':
            raise ContentSafetyError("content safety detection failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def generate_initial(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_api(client, prompt)
        return response.strip()
    except ContentSafetyError:
        print(f"content safety blocked: {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str) -> Dict:
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. You will be given two inputs: Structured Data and Generated Text, a natural language description of the data.
You are asked to evaluate the Generated Text by looking at the Structured Data.

Then, you need to answer the question: is the Generated Text a hallucination or not? If it's a hallucination, you need to find the location where the hallucination appears.
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Hallucination: yes
Location:
Text claim: 25°C
Original Data evidence: "temperature": 23.5
Text claim: at 3 PM
Original Data evidence: no supporting evidence

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: no
Location: None

Final judgment responds STRICTLY in this format:
Hallucination: [yes/no]
Location:
Text claim: "[exact text]"
Original Data evidence: "[relevant text]" 
... (only if Hallucination is yes)

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""

        response = call_api(client, prompt).strip()
        
        hallucination_types = "no"
        hallucination_details = []

        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        if lines and lines[0].startswith("Hallucination:"):
            hallucination_status = lines[0].split(":", 1)[1].strip().lower()
            hallucination_types = "yes" if hallucination_status == "yes" else "no"
            
            if hallucination_types == "yes" and len(lines) > 1 and lines[1].startswith("Location:"):
                i = 2
                while i < len(lines):
                    if lines[i].startswith("Text claim:"):
                        claim = lines[i].split(":", 1)[1].strip().strip('"')
                        i += 1
                        if i < len(lines) and lines[i].startswith("Original Data evidence:"):
                            evidence = lines[i].split(":", 1)[1].strip().strip('"')
                            hallucination_details.append({
                                "Text claim": claim,
                                "Original Data evidence": evidence or "no supporting evidence"
                            })
                        i += 1
                    else:
                        i += 1

        return {
            "hallucination_types": hallucination_types,
            "hallucination_details": hallucination_details
        }
    
    except Exception as e:
        print(f"detection error: {str(e)}")
        return {
            "hallucination_types": "no",
            "hallucination_details": []
        }

def regenerate(client: OpenAI, data: str, wrong_text: str, errors: list, reason: str) -> Optional[str]:
    try:
        errors_str = "\n".join(
            f"Discrepancy found:\n"
            f"Text claim: {error['Text claim']}\n"
            f"Original Data evidence: {error['Original Data evidence']}"
            for error in errors
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

        response = call_api(client, prompt)
        if match := re.search(r'Corrected text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
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
            
        hallucination_info = detect_hallucination(client, question, initial)
        
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "hallucination_types": hallucination_info["hallucination_types"],
            "hallucination_details": hallucination_info["hallucination_details"],
            "reference_answer": reference
        }

        if hallucination_info["hallucination_types"] == "yes" and hallucination_info["hallucination_details"]:
            if corrected := regenerate(client, question, initial, hallucination_info["hallucination_details"], ""):
                result["final_answer"] = corrected
            else:
                result["final_answer"] = initial
        else:
            result["final_answer"] = initial

        return result
        
    except ContentSafetyError:
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
                    print(f"JSON parse failed: {line}")
            except Exception as e:
                print(f"main process error: {str(e)}")

if __name__ == "__main__":
    INPUT_JSONL = "~/data/processed/complete/d2t_sampled_300.jsonl"
    OUTPUT_JSONL = "~/results/Self-Reflection/d2t/qwen_plus/d2t_qwen_plus_location_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    