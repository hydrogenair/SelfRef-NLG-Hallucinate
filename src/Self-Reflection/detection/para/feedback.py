#para feedback 300 qwenplus 7b
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
        print(f"content safety blocked: {question:[:20]}...")
        return None

def detect_hallucination(client: OpenAI, input_text: str, paraphrase: str) ->Optional[str]:
    try:
        prompt = f"""
The given task is to Evaluate whether the paraphrased text contains hallucinated content by comparison with the original text.

You are a linguistic expert in semantic analysis. You will be given two inputs: Original Text and Paraphrased Text which is the rewritten version to evaluate.
You are asked to evaluate the Paraphrased Text by looking at the Original Text.

Then, you need to answer the question: Is the generated paraphrase a hallucination or not? If it's a hallucination, you need to generate precise feedback for correction.
A paraphrased text contains hallucination IF AND ONLY IF it contains any nonsensical or factually incorrect information, or contains any additional information that cannot be supported by the original text.

Example 1 (Hallucination):
Original Text: "The European Union has 27 member states."
Paraphrase: "There are 25 countries in the EU bloc."
Feedback: the original text states '27 member states' but paraphrase says '30-year-old'. Correct the age to match the source.


Example 2 (Not Hallucination):
Original Text: "Global temperatures rose 1.1°C since 1880."
Paraphrase: "Earth's temperature has increased by 1.1 degrees Celsius since the late 19th century."
Feedback:None

Respond STRICTLY in this format:
Feedback: [Specific feedback content or None]

Now evaluate:
Original Text: {input_text}
Paraphrased Text: {paraphrase}"""

        response = call_api(client, prompt)
            
        feedback_match = re.search(r'Feedback:\s*(.+)$', response, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            return None if feedback.lower() == "none" else feedback
        return None
    except ContentSafetyError:
        return None
    
def regenerate(client: OpenAI, question: str, bad_paraphrase: str, feedback: str) -> Optional[str]:
    try:
        prompt = f"""
Original Question: {question}
Invalid Paraphrase: {bad_paraphrase}
This paraphrase contains inaccuracies. 
Critical Feedback: {feedback}
Generate a corrected version using ONLY the original question. 
Respond STRICTLY in format:
Corrected Paraphrase: [your text here]
"""

        response = call_api(client, prompt)
        return response.strip()
    except ContentSafetyError:
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"]
        
        # Generate initial
        initial = generate_initial(client, question)
        if initial is None:
            return None
            
        # Detect hallucinations
        feedback = detect_hallucination(client, question, initial)
            
        # 修改后的输出逻辑
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "feedback": feedback,
            "reference_answer": reference
        }

        if feedback:
            if corrected := regenerate(client, question, initial, feedback):
                result["final_answer"] = corrected
            else:
                return None  # 安全检测失败时跳过
        else:
            result["final_answer"] = initial  # 保持初始摘要

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
    OUTPUT_JSONL = "~/results/Self-Reflection/para/llama/para_llama_feedback_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    