#d2t feedback 300 llama
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

Then, you need to answer the question: is the Generated Text a hallucination or not? If it's a hallucination, you need to generate precise feedback for correction.
A generated text contains hallucination if and only if it contradicts the source data or contains nonsensical or factually incorrect information.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Feedback: the original data states '"temperature": 23.5' but the text says '25°C'. Correct the age to match the source.


Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Feedback:None



Respond STRICTLY in this format:
Feedback: [Specific feedback content or None]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
"""

        response = call_api(client, prompt).strip()
        print(response)
        return response

    except ContentSafetyError:
        return None
    
    
def regenerate(client: OpenAI, data: str, wrong_text: str, feedback: str) -> Optional[str]:
    try:
        prompt = f"""
Request: {data}
Invalid text: {wrong_text}
This generated text contains inaccuracies. 
Critical Feedback: {feedback}
Generate a corrected version using ONLY information from the Structured Data. 
Respond STRICTLY in this format: 
Corrected text: [your text here] 
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
        feedback_response = detect_hallucination(client, question, initial)
        print(feedback_response)
        
        if feedback_response and feedback_response.lower().startswith("feedback:"):
            feedback_content = feedback_response.split(":", 1)[1].strip()
        else:
            feedback_content = "None"
            
        result = {
            "input": input_text,
            "question": question,
            "initial_answer": initial,
            "feedback": feedback_content,
            "reference_answer": reference

        }

        if feedback_content != "None":
            if corrected := regenerate(client, question, initial, feedback_content):
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
    OUTPUT_JSONL = "~/results/Self-Reflection/d2t/qwen0.5b/d2t_qwen0.5b_feedback_300.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    