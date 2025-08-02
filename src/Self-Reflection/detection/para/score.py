#para score llama
HALLUCINATION_THRESHOLD = 5
import json
import re
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

class ContentSafetyError(Exception):
    """Content safety inspection exception"""

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
            raise ContentSafetyError("Content safety check failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def generate_initial_paraphrase(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_api(client, prompt)
        return response.strip() if response else None
    except ContentSafetyError:
        print(f"Content blocked: {question[:20]}...")
        return None

def detect_paraphrase_hallucination(client: OpenAI, input_text: str, paraphrase: str) -> Dict:
    try:
        prompt = f"""
The given task is to evaluate the faithfulness of the paraphrased text by comparing it with the original text.

You are a linguistic expert in semantic analysis . You will be given two inputs: Original Text and Paraphrased Text which is the rewritten version to evaluate.
You are asked to evaluate paraphrased text based on the original content.

Then, you need to answer the question: is the Paraphrased Text a hallucination or not?
A paraphrased text contains hallucination IF it contains any nonsensical or factually incorrect information, or contains any additional information that cannot be supported by the original text.

Your mission is to Score the paraphrase quality from 1-10 based on faithfulness to the original text.
1-3: Severe hallucination (key facts altered/missing)
4-6: Partial hallucination (minor inaccuracies)
7-10: Factually consistent (preserves core meaning)

Example 1:
Original: "The EU has 27 members."
Paraphrase: "There are 25 EU countries."
Score: 3

Example 2:
Original: "Temperatures rose 1.1°C since 1880."
Paraphrase: "1.1°C increase since late 19th century."
Score: 9

Respond STRICTLY in format:
Score: [1-10]

Now evaluate:
Original Text: {input_text}
Paraphrased Text: {paraphrase}"""
        
        response = call_api(client, prompt)
        score_match = re.search(r'Score:\s*(\d+)', response)
        return {"score": int(score_match.group(1)) if score_match else 1}
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return {"score": 1}

def regenerate_paraphrase(client: OpenAI, question: str, bad_paraphrase: str, score: int) -> Optional[str]:
    try:
        prompt = f"""Initial paraphrase scored {score}/10 (threshold {HALLUCINATION_THRESHOLD}), which means the initial paraphrase contains inaccuracies or hallucinations.
Improve faithfulness to the original question:

Original: {question}
Bad Paraphrase: {bad_paraphrase}
Respond STRICTLY in the format:
Paraphrased Text: [your text here]"""
        
        response = call_api(client, prompt)
        if match := re.search(r'Paraphrased Text:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        return response.strip()  # Fallback processing
    except ContentSafetyError:
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"][0]  # Take the first reference answer

        initial = generate_initial_paraphrase(client, question)
        if not initial:
            return None
    
        detection = detect_paraphrase_hallucination(client, input_text, initial)
        needs_correction = detection["score"] < HALLUCINATION_THRESHOLD
        
        final_answer = initial
        if needs_correction:
            if corrected := regenerate_paraphrase(client, question, initial, detection["score"]):
                final_answer = corrected
        
        return {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "score": detection["score"],
            "final_answer": final_answer,
            "reference_answer": reference
        }
    except ContentSafetyError:
        return {
            "input": input_text,
            "question": question,
            "initial_answer": "[Blocked]",
            "hallucination": "yes",
            "final_answer": "[Blocked]",
            "reference_answer": reference
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
    INPUT_JSONL = "data/processed/complete/para_sampled_300.jsonl"
    OUTPUT_JSONL = "results/Self-Reflection/para/llama/para_llama_score_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)