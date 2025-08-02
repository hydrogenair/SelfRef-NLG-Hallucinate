'''{
  "hallucination_types": " Contradictory,Unverifiable",
  "hallucination_details": [
    {
      "types": ["Contradictory"],
      "Text claim": "25°C",
      "Original Data evidence": ""temperature": 23.5'"
    },
    {
      "types": ["Unverifiable"],
      "Text claim": "at 3 PM",
      "Original Data evidence": "no supporting evidence"
    }
  ]
} '''
#d2t position+type 300 llama
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
            model="qwen2.5-0.5b-instruct",
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

Then, you need to answer the question: is the Generated Text a hallucination or not?if it's a hallucination, you need to find the location where the hallucination appears and the type of the hallucination.
The possible types of hallucination are:
1. Contradictory Hallucination: The text includes information that directly contradicts the original article.
2. Unverifiable Hallucination: The text includes information that cannot be supported or verified by the original article.
3. No Fact: The point does not contain any factual information to be judged.
If the text is fully consistent and all information is supported by the article, the hallucination type is None.
For each claim in the text, determine if it falls into any of the above categories. After evaluating all claims, provide the final judgment by listing all applicable hallucination types separated by commas if multiple. If there are none, state None.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Hallucination: Contradictory,Unverifiable
Location:
types: Contradictory
Text claim: 25°C
Original Data evidence: "temperature": 23.5
types: Unverifiable
Text claim: at 3 PM
Original Data evidence: no supporting evidence

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: None
Location: None

Final Judgement and Type can only select one or more of the following answers: None,Contradictory,Unverifiable,No Fact.
Responds STRICTLY in this format:
Hallucination: [Your Final Judgement]
Location:
types: [Type]
Text claim: "[exact text]"
Original Data evidence: "[relevant text]" 
... (only if Hallucination is Yes)

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""

        response = call_api(client, prompt).strip()
        
        hallucination_types = []
        hallucination_details = []
        
        type_line = next((line for line in response.split('\n') if line.startswith("Hallucination:")), None)
        if type_line:
            raw_types = type_line.split(":", 1)[1].strip()
            if raw_types != "None":
                hallucination_types = list(set([t.strip() for t in raw_types.split(",")]))
        
        current_error = None
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("types:"):
                if current_error:
                    hallucination_details.append(current_error)
                current_error = {
                    "types": [t.strip() for t in line.split(":",1)[1].split(",")],
                    "Text claim": None,
                    "Original Data evidence": None
                }
            elif line.startswith("Text claim:") and current_error:
                current_error["Text claim"] = line.split(":",1)[1].strip().strip('"')
            elif line.startswith("Original Data evidence:") and current_error:
                current_error["Original Data evidence"] = line.split(":",1)[1].strip().strip('"')
                hallucination_details.append(current_error)
                current_error = None
        
        if current_error:
            hallucination_details.append(current_error)
        
        final_types = "None" if not hallucination_types else ",".join(sorted(hallucination_types))
        
        return {
            "hallucination_types": final_types,
            "hallucination_details": hallucination_details
        }
    
    except Exception as e:
        print(f"detection error: {str(e)}")
        return {
            "hallucination_types": "None",
            "hallucination_details": []
        }
    
def regenerate(client: OpenAI, data: str, wrong_text: str, errors: list) -> Optional[str]:
    try:
        errors_str = []
        for error in errors:
            detail = (
                f"Type: {', '.join(error['types'])}\n"
                f"Text claim: {error['Text claim']}\n"
                f"Original Data evidence: {error['Original Data evidence'] or 'No supporting evidence'}"
            )
            errors_str.append(detail)
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

        hallucination_result = detect_hallucination(client, question, initial)
        
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "reference_answer": reference,
            "hallucination_types": hallucination_result["hallucination_types"],
            "hallucination_details": hallucination_result["hallucination_details"]
        }

        if hallucination_result["hallucination_types"] != "None" and hallucination_result["hallucination_details"]:
            if corrected := regenerate(client, question, initial, hallucination_result["hallucination_details"]):
                result["final_answer"] = corrected
            else:
                return None
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
    OUTPUT_JSONL = "~/results/Self-Reflection/d2t/qwen0.5b/d2t_qwen0.5b_location_category_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    