#none-para
import json
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

def regenerate_final_answer(client: OpenAI, input_text: str, initial_answer: str) -> Optional[str]:
    try:
        prompt = f"""Generate a revised final answer based on the original input and question. 
        The initial answer may contain inaccuracies or hallucinations.
        Original Input:{input_text}
        Initial Answer (May contain errors):{initial_answer}

Generate the revised final answer:"""
        
        response = call_api(client, prompt)
        return response.strip() if response else None
    except ContentSafetyError:
        return None
    except Exception as e:
        print(f"Regeneration error: {str(e)}")
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"][0]
        
        # 生成初始答案
        initial = generate_initial_paraphrase(client, question)
        if not initial:
            return None
            
        # 直接生成最终答案
        final_answer = regenerate_final_answer(client, input_text, initial)
        
        return {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "final_answer": final_answer if final_answer else "[Blocked]",
            "reference_answer": reference
        }
    except ContentSafetyError:
        return {
            "input": input_text,
            "question": question,
            "initial_answer": "[Blocked]",
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
    main(
        "/data/processed/complete/para_sampled_50.jsonl",
        "/results/Self-Reflection/para/llama/para_llama_no_300.jsonl"
    )