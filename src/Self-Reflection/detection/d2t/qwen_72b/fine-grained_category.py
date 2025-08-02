#d2t fine category 300 llama
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
            model="qwen2.5-72b-instruct",
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

Then, you need to answer the question: is the Generated Text a hallucination or not?
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.
If it's a hallucination, you need to categorize the hallucinaition into one of the following categories : 
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

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Hallucination: None

final Judgement can only select one or more of the following answers: NAME, NUMBER, WORD, CONTEXT, OTHER, NOT CHECKABLE, None,
and can only be Responded STRICTLY in the format:
Hallucination:[your Final Judgement]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""

        response = call_api(client, prompt).strip()
        match = re.search(r'Hallucination:\s*([^\n]+)', response)
        if not match:
            return 'None'
        
        types_str = match.group(1).strip()
        valid_types = []
        allowed = {'NAME', 'NUMBER', 'WORD', 'CONTEXT', 'OTHER', 'NOT CHECKABLE','None'}
        
        for t in re.split(r',\s*', types_str):
            clean_t = re.sub(r'\W+', '', t.split()[0] if t else '').strip()
            if clean_t == 'NoFact':
                clean_t = 'No Fact'
            if clean_t == 'NONE':
                clean_t='None'
            if clean_t in allowed:
                valid_types.append(clean_t)
        
        if 'None' in valid_types:
            return 'None'
        return ','.join(sorted(set(valid_types))) if valid_types else 'None'
    except ContentSafetyError:
        return None
    except Exception as e:
        print(f"detection error: {str(e)}")
        return 

def regenerate_answer(client: OpenAI, data: str, wrong_text: str, hallucination_types:str) -> Optional[str]:
    try:
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
     
        This Generated Text contains {hallucination_types} hallucinations.
        Generate a corrected version using ONLY information from the Structured Data. 
        Respond STRICTLY in this format: 
        Corrected text: [your text here] 
"""
        
        response = call_api(client, prompt)
        if match := re.search(r'Corrected text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        return response
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
            
        hallucination_types = detect_hallucination(client, question, initial)
        if hallucination_types is None:
            return None
            
        has_hallu = hallucination_types != 'None'
        
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "hallucination": "yes" if has_hallu else "no",
            "hallucination_types": hallucination_types,
            "reference_answer": reference
        }

        if has_hallu:
            relevant_types = [t for t in hallucination_types.split(',') 
                            if t in ['NAME', 'NUMBER', 'WORD', 'CONTEXT', 'OTHER', 'NOT CHECKABLE']]
            if relevant_types:
                if corrected := regenerate_answer(client, question, initial, ', '.join(relevant_types)):
                    result["final_answer"] = corrected
                else:
                    return None
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
    OUTPUT_JSONL = "~/results/Self-Reflection/d2t/qwen_72b/d2t_qwen_72b_fine-grained_category_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    