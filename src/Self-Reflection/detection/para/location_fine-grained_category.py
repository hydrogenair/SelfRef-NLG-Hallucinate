#para position+type 300 llama 
'''{
  "hallucination_types": "NUMBER,OTHER",
  "hallucination_details": [
    {
      "types": ["NUMBER"],
      "Paraphrase claim": "30-year-old",
      "Original Text evidence": "Article states '29-year-old'"
    },
    {
      "types": ["OTHER"],
      "Paraphrase claim": "and the United States",
      "Original Text evidence": "no supporting evidence"
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



def generate_initial(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_api(client, prompt)
        return  response.strip()
    except ContentSafetyError:
        print(f"Content safety blocked: first 20 characters of article {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, input_text: str, paraphrase: str) -> tuple:
    try:
        prompt = f"""
The given task is to Evaluate whether the paraphrased text contains hallucinated content by comparison with the original text.

You are a linguistic expert in semantic analysis. You will be given two inputs: Original Text and Paraphrased Text which is the rewritten version to evaluate.
You are asked to evaluate the Paraphrased Text by looking at the Original Text.

Then, you need to answer the question: Is the generated paraphrase a hallucination or not? if it's a hallucination, you need to find the location where the hallucination appears and the type of the hallucination.
A paraphrased text contains hallucination IF AND ONLY IF it contains any nonsensical or factually incorrect information, or contains any additional information that cannot be supported by the original text.
Hallucinations are categorized as follows:
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
Hallucination: NUMBER,OTHER
Location:
types: NUMBER
Paraphrase claim: "25 countries" 
Original Text evidence: "27 member states" 
Location:
types: OTHER
Paraphrase claim: "and the United States" 
Original Text evidence: "no supporting evidence" 

Example 2 (Not Hallucination):
Original Text: "Global temperatures rose 1.1°C since 1880."
Paraphrase: "Earth's temperature has increased by 1.1 degrees Celsius since the late 19th century."
Hallucination: None
Location: None

Responds STRICTLY in this format:
Final Judgement and Type can only select one or more of the following answers: NAME,NUMBER,WORD,CONTEXT,OTHER,NOT CHECKABLE,None
Hallucination: [Your Final Judgement]
Location:
types:[Type]
Paraphrase claim: "[exact text]"
Original Text evidence: "[relevant text]" 
... (only if Hallucination is Yes)

Now evaluate:
Original Text: {input_text}
Paraphrased Text: {paraphrase}"""

        response = call_api(client, prompt).strip()
        
        hallucination_types = []
        errors = []
        
        # Extract main hallucination types
        type_match = re.search(r'Hallucination:\s*([^\n]+)', response)
        if type_match:
            raw_types = type_match.group(1).strip()
            hallucination_types = [t.strip() for t in raw_types.split(",") if t.strip() in ['NAME','NUMBER','WORD','CONTEXT','OTHER','NOT CHECKABLE']]

        # Parse detailed errors
        current_error = None
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("types:"):
                if current_error is None:
                    current_error = {"types": [], "claim": None, "evidence": None}
                current_error["types"] = [t.strip() for t in line.split(":", 1)[1].strip().split(",")]
            elif line.startswith("Paraphrase claim:"):
                if current_error is None:
                    current_error = {"types": [], "claim": None, "evidence": None}
                current_error["claim"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("Original Text evidence:") and current_error:
                current_error["evidence"] = line.split(":", 1)[1].strip().strip('"')
                errors.append(current_error)
                current_error = None

        final_types = "None" if not hallucination_types else ",".join(sorted(hallucination_types))
        return (final_types, errors)
    except Exception as e:
        print(f"Detection failed: {str(e)}")
        return ("None", [])

def regenerate(client: OpenAI, question: str, bad_paraphrase: str, errors: list) -> Optional[str]:
    try:
        errors_str = []
        for error in errors:
            detail = (
                f"Type: {', '.join(error['types'])}\n"
                f"Paraphrase claim: {error['claim']}\n"
                f"Original Text evidence: {error['evidence'] or 'No supporting evidence'}"
            )
            errors_str.append(detail)
        prompt = f"""
Original Question: {question}
Invalid Paraphrase: {bad_paraphrase}
This paraphrase contains inaccuracies. The following discrepancies were found between the paraphrase and the original text:
{errors_str}

Generate a corrected version using ONLY the original question. 
Respond STRICTLY in format:
Corrected Paraphrase: [your text here]
"""

        response = call_api(client, prompt)
        if not response:
            return None
            
        # Enhanced format matching logic
        if corrected := re.search(r'Corrected Paraphrase:\s*(.+)$', response, re.DOTALL):
            return corrected.group(1).strip()
        return response.strip()  # Fallback processing
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
        hallu_types, errors = detect_hallucination(client, input_text, initial)
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "reference_answer": reference,
            "hallucination_types": hallu_types,
            "hallucination_details": [
                {
                    "types": error.get("types", []),
                    "Paraphrase claim": error.get("claim", ""),
                    "Original Text evidence": error.get("evidence") or "no supporting evidence"
                } for error in errors
            ]
        }
        if hallu_types != "None" and errors:
            if corrected := regenerate(client, question, initial, errors):
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
                print(f"JSON parsing failed: {line}")
            except Exception as e:
                print(f"Main process exception: {str(e)}")

if __name__ == "__main__":
    INPUT_JSONL = "data/processed/complete/para_sampled_300.jsonl"
    OUTPUT_JSONL = "results/Self-Reflection/para/llama/para_llama_loc-fine-grained-type_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    