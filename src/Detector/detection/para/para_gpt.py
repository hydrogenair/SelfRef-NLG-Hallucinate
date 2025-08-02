
import json
import re
from typing import Dict, List
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
            model="gpt-4o",
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

def detect_paraphrase_hallucination(client: OpenAI, input_text: str, paraphrase: str, reference_answer: str) -> Dict[str, str]:
    try:
        prompt = f"""
The given task is that Evaluate whether the paraphrased text contains hallucinated content by comparison with the original text.

You are a linguistic expert in semantic analysis. You will be given three inputs:Original Text、Paraphrased Text (rewritten version to evaluate) and Reference Answer (dataset’s standard factual answer for comparison).
You are asked to evaluate the Paraphrased Text looking at the Original Text and Reference Answer.

Then, you need to answer the question: is the Paraphrased Text a hallucination or not?
A paraphrased text contains hallucination IF AND ONLY IF it contains any nonsensical or factually incorrect information, or contains any additional information that cannot be supported by either the original text.

Example 1 (Hallucination):
Original Text: "The European Union has 27 member states."
Reference Answer: "The EU comprises 27 member countries."
Paraphrase: "There are 25 countries in the EU bloc."
Hallucination: Yes
Reason: Incorrect member count (25 vs 27 in Original Text and Reference Answer).

Example 2 (Not Hallucination):
Original Text: "Global temperatures rose 1.1°C since 1880."
Reference Answer: "Earth’s average temperature increased by 1.1°C since the late 1800s."
Paraphrase: "Earth's temperature has increased by 1.1 degrees Celsius since the late 19th century."
Hallucination: No
Reason: Facts align with both Original Text and Reference Answer.

Respond STRICTLY in this format:
Hallucination: [yes/no]
Reason: [Your explanation]

Now evaluate this case:
Original Text: {input_text}
Reference Answer: {reference_answer}
Paraphrased Text: {paraphrase}"""
        
        response = call_gpt_api(client, prompt)
        
        # Extract judgment and reason
        judgment_match = re.search(r'Hallucination:\s*(yes|no)', response, re.IGNORECASE)
        reason_match = re.search(r'Reason:\s*(.+)', response, re.IGNORECASE)
        
        return {
            "gpt_final": judgment_match.group(1).capitalize() if judgment_match else "No",
            "gpt_reason": reason_match.group(1).strip() if reason_match else "No reason provided"
        }
    except Exception as e:
        print(f"detection failed: {input_text[:50]}... - {str(e)}")
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
            for para in initial_answers:
                result = detect_paraphrase_hallucination(
                    client=client,
                    input_text=data["input"],
                    paraphrase=para,
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
                "gpt_reason": judgments[0]["gpt_reason"] if judgments else "No paraphrase provided"
            }
            
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "~/data/processed/detector/para_sampled_50.jsonl"
    output_file = "~/results/Detector/para/para_gpt_50.jsonl"
    
    process_paraphrase_data(input_file, output_file)