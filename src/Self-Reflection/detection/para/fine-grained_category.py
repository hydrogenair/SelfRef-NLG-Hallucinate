#para 细粒度category 300 qwenplus 
import re
import os
import json
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

HALLUCINATION_THRESHOLD = 5

class ContentSafetyError(Exception):
    """content safety detection exception"""

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

def detect_hallucination(client: OpenAI, input_text: str, paraphrase: str) -> Dict:
    try:
        prompt = f"""
The given task is to Evaluate whether the paraphrased text contains hallucinated content by comparison with the original text.

You are a linguistic expert in semantic analysis. You will be given two inputs: Original Text and Paraphrased Text which is the rewritten version to evaluate.
You are asked to evaluate the Paraphrased Text by looking at the Original Text.

Then, you need to answer the question: is the Paraphrased Text a hallucination or not?
A paraphrased text contains hallucination IF AND ONLY IF it contains any nonsensical or factually incorrect information, or contains any additional information that cannot be supported by the original text.

If it's a hallucination, you need to categorize the hallucinaition into one of the following categories : 
1.  Incorrect named entity(NAME): This includes people, places, organizations, and days of the week.
2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)
If the paraphrase is fully consistent and all information is supported by the article, the hallucination type is None.
For each claim in the paraphrase, determine if it falls into any of the above categories. After evaluating all claims, provide the final judgment by listing all applicable hallucination types separated by commas if multiple. If there are none, state None.

Example 1 (Hallucination):
Original Text: "The European Union has 27 member states."
Paraphrase: "There are 25 countries in the EU bloc."
Hallucination: NUMBER

Example 2 (Not Hallucination):
Original Text: "Global temperatures rose 1.1°C since 1880."
Paraphrase: "Earth's temperature has increased by 1.1 degrees Celsius since the late 19th century."
Hallucination: None

Final Judgement can only select one or more of the following answers:NAME,NUMBER,WORD,CONTEXT,OTHER,NOT CHECKABLE,None,
and can only be Responded STRICTLY in the format:
Hallucination:[your Final Judgement]

Now evaluate:
Original Text: {input_text}
Paraphrased Text: {paraphrase}"""

        response = call_api(client, prompt).strip()
        match = re.search(r'Hallucination:\s*([^\n]+)', response)
        if not match:
            return None
        
        types_str = match.group(1).strip()
        valid_types = []
        allowed = {'NAME', 'NUMBER', 'WORD', 'CONTEXT', 'OTHER', 'NOT CHECKABLE','None'}
        
        # Clean and validate types
        for t in re.split(r',\s*', types_str):
            clean_t = re.sub(r'\W+', '', t.split()[0] if t else '').strip()
            if clean_t == 'NoFact':
                clean_t = 'No Fact'
            if clean_t in allowed:
                valid_types.append(clean_t)
        
        # Priority to None
        if 'None' in valid_types:
            return 'None'
        return ','.join(sorted(set(valid_types))) if valid_types else 'None'
    except ContentSafetyError:
        return None
    except Exception as e:
        print(f"detection error: {str(e)}")
        return 
    
def regenerate(client: OpenAI, question: str, bad_paraphrase: str, hallucination_types: str) -> Optional[str]:
    try:
        prompt = f"""

Original Question: {question}
Invalid Paraphrase: {bad_paraphrase}

Hallucinations are categorized as follows:
        1.  Incorrect named entity(NAME): This includes people, places, organizations, and days of the week.
        2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
        3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
        4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
        5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
        6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)
        
This Paraphrased Text contains {hallucination_types} hallucinations. Generate a corrected version using ONLY the original question. 
Respond STRICTLY in format:
Corrected Paraphrase: [your text here]
"""
        response = call_api(client, prompt)
            
        if match := re.search(r'Corrected Paraphrase:\s*(.+)$', response, re.DOTALL):
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
            
        hallucination_types = detect_hallucination(client, input_text, initial)
        if hallucination_types is None:
            return None
            
        # Determine if contains any hallucination
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
                            if t in ['NAME','NUMBER','WORD','CONTEXT','OTHER','NOT CHECKABLE']]
            if relevant_types:
                if corrected := regenerate(client, question, initial, ', '.join(relevant_types)):
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
    INPUT_JSONL = "~/data/processed/complete/para_sampled_300.jsonl"
    OUTPUT_JSONL = "~/results/Self-Reflection/para/llama/para_llama_fine-grained_category_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    