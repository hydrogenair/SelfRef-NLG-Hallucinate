import json
import re
import os
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

class ContentSafetyError(Exception):
    """Content safety inspection exception"""

def create_deepseek_client():
    return OpenAI(
        api_key="",
        base_url="",
    )

def create_qwen_client():
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

def generate_initial_paraphrase(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_qwen_api(client, prompt)
        return response.strip() if response else None
    except ContentSafetyError:
        print(f"Content blocked: {question[:20]}...")
        return None

def detect_paraphrase_hallucination(client: OpenAI, input_text: str, paraphrase: str) -> Optional[bool]:
    try:
        prompt = f"""
The given task is that Evaluate whether the paraphrased text contains hallucinated content by comparison with the original text.

You are a linguistic expert in semantic analysis . You will be given two inputs: Original Text and Paraphrased Text which is the rewritten version to evaluate.
You are asked to evaluate the Paraphrased Text looking at the Original Text.

Then, you need to answer the question: is the Paraphrased Text a hallucination or not?
A paraphrased text contains hallucination IF AND ONLY IF it contains any nonsensical or factually incorrect information, or contains any additional information that cannot be supported by the original text.


Example 1 (Hallucination):
Original Text: "The European Union has 27 member states."
Paraphrase: "There are 25 countries in the EU bloc."
Analysis: Violates condition 1 (incorrect number)
Example 2 (Not Hallucination):
Original Text: "Global temperatures rose 1.1°C since 1880."
Paraphrase: "Earth's temperature has increased by 1.1 degrees Celsius since the late 19th century."
Analysis: Preserves all factual elements

Now evaluate this case:
Original Text: {input_text}
Paraphrased Text: {paraphrase}
Final judgment responds STRICTLY in this format: Hallucination: [Yes/No]"""
        response = call_deepseek_api(client, prompt).strip()
        return response.lower().startswith("hallucination: yes")
    except ContentSafetyError:
        return None
    except Exception as e:
        print(f"Detection Error: {str(e)}")
        return True

def regenerate_paraphrase(client: OpenAI, question: str, bad_paraphrase: str) -> Optional[str]:
    try:
        prompt = f"""This paraphrase contains invalid additions. Generate a corrected version using ONLY the original question. 
Respond STRICTLY in format:
Corrected Paraphrase: [your text here]

Original Question: {question}
Invalid Paraphrase: {bad_paraphrase}"""
        
        response = call_qwen_api(client, prompt)
        if match := re.search(r'Corrected Paraphrase:\s*(.+)$', response, re.DOTALL):
            return match.group(1).strip()
        return None
    except ContentSafetyError:
        return None

def process_entry(qwen_client: OpenAI, deepseek_client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"]  # Take the first reference answer
        
        initial = generate_initial_paraphrase(qwen_client, question)
        if not initial:
            return None
            
        has_hallucination = detect_paraphrase_hallucination(deepseek_client, input_text, initial)
        
        final_answer = initial
        if has_hallucination:
            if corrected := regenerate_paraphrase(qwen_client, question, initial):
                final_answer = corrected
            else:
                final_answer = "[Content Blocked]"
        
        return {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "hallucination": "yes" if has_hallucination else "no",
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
    INPUT_JSONL = "data/processed/detector/para_sampled_50.jsonl"
    OUTPUT_JSONL = "results/External Annotation/para/para_deepseek_72b_binary_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)