
import json
import re
from typing import Dict, List,Optional
from openai import OpenAI, APIError

class ContentSafetyError(Exception):
    """content safety detection failed"""

def create_client():
    return OpenAI(
        api_key="",
        base_url="",
    )

def call_gpt_api(client: OpenAI, prompt: str) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model="qwen-max",
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

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str,reference_answer: str) -> Optional[bool]:
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. 
You will be given three inputs: Structured Data ,Generated Text which is a natural language description of the data and and Reference Answer (dataset’s standard factual answer for comparison).
You are asked to evaluate the Generated Text by looking at the Structured Data.

Then, you need to answer the question: is the Generated Text a hallucination or not?
A generated text contains hallucination if and only if it icontradicts the source data,or any nonsensical or factually incorrect information.

Example 1 (Hallucination):
Structured Data: {{ "temperature": 23.5, "humidity": 68 }}
Generated Text: "The sensor recorded 25°C with 70% humidity at 3 PM"
Reason: Violates condition 1 (incorrect values)

Example 2 (No Hallucination):
Structured Data: {{ "sales": {{ "Q1": 1500000, "Q2": 1800000 }} }}
Generated Text: "Q2 sales showed 20% growth compared to Q1"
Reason: Matches data (1.8M vs 1.5M = actual 20% increase)

Final judgment responds STRICTLY in this format: 
Hallucination: [Yes/No]
Reason: [Your explanation]

Now evaluate this case:
Structured Data: {input_data}
Generated Text: {generated_text}
Reference Answer: {reference_answer}"""
        
        response = call_gpt_api(client, prompt)
        
        # Extract judgment and reason
        judgment_match = re.search(r'Hallucination:\s*(yes|no)', response, re.IGNORECASE)
        reason_match = re.search(r'Reason:\s*(.+)', response, re.IGNORECASE)
        
        return {
            "gpt_final": judgment_match.group(1).capitalize() if judgment_match else "No",
            "gpt_reason": reason_match.group(1).strip() if reason_match else "No reason provided"
        }
    except Exception as e:
        print(f"detection failed: {input_data[:50]}... - {str(e)}")
        return {"gpt_final": "No", "gpt_reason": f"Error in detection: {str(e)}"}

def process_paraphrase_data(input_path: str, output_path: str):
    client = create_client()
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            
            initial_answers = data.get("initial_answer", [])
            if not isinstance(initial_answers, list):
                initial_answers = [initial_answers] if initial_answers else []
            
            judgments = []
            for ans in initial_answers:
                result = detect_hallucination(
                    client=client,
                    input_data=data["input"],
                    generated_text=ans,
                    reference_answer=data["reference_answer"]
                )
                judgments.append(result)
            
            output_data = {
                "question": data.get("question", ""),
                "input": data.get("input", ""),
                "initial_answer": data.get("initial_answer", []),
                "hallucination": data.get("hallucination", ""),
                "final_answer": data.get("final_answer", ""),
                "reference_answer": data.get("reference_answer", ""),
                "gpt_final": judgments[0]["gpt_final"] if judgments else "No",
                "gpt_reason": judgments[0]["gpt_reason"] if judgments else "No text provided"
            }
            
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "~/data/processed/detector/d2t_sampled_50.jsonl"
    output_file = "~/results/Detector/d2t/d2t_qwen_50.jsonl"
    
    process_paraphrase_data(input_file, output_file)