#### d2t no
import json
import re
import os
from typing import Dict, Optional
from openai import OpenAI
from openai import APIError

class ContentSafetyError(Exception):
    """Content safety inspection exception"""

def create_client():
    return OpenAI(
        api_key="",
        base_url="",
    )

def call_gpt_api(client: OpenAI, prompt: str) -> Optional[str]:
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

def detect_hallucination(client: OpenAI, input_data: str, generated_text: str, reference_answer: str) -> Dict[str, str]:
   
    try:
        prompt = f"""
The given task is to evaluate whether the generated text contains hallucinated content by comparison with the original structured data.

You are a data accuracy expert in semantic validation. 
You will be given three inputs: Structured Data ,Generated Text which is a natural language description of the data and and Reference Answer (dataset's standard factual answer for comparison).
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
Reference Answer: {reference_answer}
   """
        
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

def main(input_file: str):
    client = create_client()
    total_initial_hallucination_count = 0
    total_final_hallucination_count = 0
    total_questions = 0
    
    base_name = os.path.basename(input_file)
    output_name = f"eval_{base_name.split('_')[2]}.json"
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    
    results = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                total_questions += 1
                ref_answer = entry.get("reference_answer", "")
                
                result_entry = {
                    "input": entry["input"],
                    "initial_answer": entry["initial_answer"],
                    "final_answer": entry["final_answer"],
                    "reference_answer": ref_answer
                }
                
                # 检测 initial_para 的幻觉
                if entry["initial_answer"] != "[Blocked]":
                    initial_result = detect_hallucination(
                        client, 
                        entry["input"],
                        entry["initial_answer"],
                        ref_answer
                    )
                    if initial_result["gpt_final"] == "Yes":
                        total_initial_hallucination_count += 1
                    result_entry["initial_eval"] = initial_result
                
                # 检测 final_para 的幻觉
                if entry["final_answer"] != "[Blocked]":
                    final_result = detect_hallucination(
                        client,
                        entry["input"],
                        entry["final_answer"],
                        ref_answer
                    )
                    if final_result["gpt_final"] == "Yes":
                        total_final_hallucination_count += 1
                    result_entry["final_eval"] = final_result
                
                results.append(result_entry)
                        
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
            except KeyError as e:
                print(f"Missing key in JSON: {e}")
            except Exception as e:
                print(f"Main process error: {str(e)}")

    initial_hallucination_rate = (total_initial_hallucination_count / total_questions) * 100 if total_questions > 0 else 0
    final_hallucination_rate = (total_final_hallucination_count / total_questions) * 100 if total_questions > 0 else 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump({
            "summary": {
                "total_questions": total_questions,
                "initial_hallucination_count": total_initial_hallucination_count,
                "initial_hallucination_rate": initial_hallucination_rate,
                "final_hallucination_count": total_final_hallucination_count,
                "final_hallucination_rate": final_hallucination_rate
            },
            "detailed_results": results
        }, outfile, ensure_ascii=False, indent=2)

    print(f"\nResults for {base_name}:")
    print("=" * 50)
    print("initial hallucination statistics:")
    print(f"total questions: {total_questions}")
    print(f"hallucination count: {total_initial_hallucination_count}")
    print(f"hallucination rate: {initial_hallucination_rate:.2f}%\n")

    print("final hallucination statistics:")
    print(f"total questions: {total_questions}")
    print(f"hallucination count: {total_final_hallucination_count}")
    print(f"hallucination rate: {final_hallucination_rate:.2f}%")
    print(f"\ndetailed results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide an input file path")