#sum location+ fine-grained category 300 llama
'''{
  "hallucination_types": "Contradictory,Unverifiable",
  "hallucination_details": [
    {
      "type": ["Contradictory"],
      "claim": "30-year-old",
      "evidence": "Article states '29-year-old'"
    },
    {
      "type": ["Unverifiable"],
      "claim": "3-2 score",
      "evidence": "No supporting evidence"
    }
  ]
} '''
import json
import re
import os
from typing import Dict, Optional, List, Tuple
from openai import OpenAI
from openai import APIError

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

def generate_initial_summary(client: OpenAI, question: str) -> Optional[str]:
    try:
        prompt = f"""{question}"""
        
        response = call_api(client, prompt)
        return response.strip()
    except ContentSafetyError:
        print(f"Content safety blocked: first 20 characters of article {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, article: str, summary: str) -> Tuple[Optional[str], Optional[List[Dict]], Optional[str]]:
    try:
        prompt = f""" 
The given task is to evaluate whether the generated summary contains hallucinated content by comparing it with the original article.

You are a linguist. Your task is to strictly compare the summary with the original article, identifying any discrepancies between the two. Focus on factual consistency and logical derivation.
You will be given two inputs: Original Request including the original Article and Generated Summary (CandidateSummary) to evaluate.

Then, you need to answer the question: Is the generated summary a hallucination or not? if it's a hallucination, you need to find the location where the hallucination appears and the type of these hallucinations.
Hallucinations are categorized as follows:
1.  Incorrect named entity(NAME): This includes people, places, organizations, and days of the week.
2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)
If the summary is fully consistent and all information is supported by the article, the hallucination type is None.
For each claim in the summary, determine if it falls into any of the above categories. After evaluating all claims, provide the final judgment by listing all applicable hallucination types separated by commas if multiple. If there are none, state None.


Example 1 (Hallucination):
Request: "Generate an appropriate summary for the given text such that it includes the main topic of the text. By. Rajvir Rai. Lukasz Fabianski has admitted he has already forgotten his time as an Arsenal player. The 29-year-old spent seven years at the Emirates before joining Swansea on a free transfer earlier this summer. Fabianski only started one league game last season - though he did start the FA Cup final victory over Hull in May - and has revealed he has already put his time at Arsenal behind. Hoping to kick on: Lukasz Fabianski has left Arsenal for Swansea in search of more game time. No chance: Lukasz Fabianski is left helpless as Villarreal scores in a pre-season friendly. Glory boy: Lukasz Fabianski celebrates winning the FA Cup with Arsenal in May. 'It was not hard leaving Arsenal. I have already kind of forgotten about it. It's behind me,' he told The Sun. 'I had many wonderful years there. But I am a Swansea player now and want to focus on that.'."
Summary: "Keeper spent seven years at Arsenal before joining Swans in the summer .30-year-old only played one league game last season .\nFabianski did start Arsenal's 3-2 FA Cup final win over Hull in May."
Hallucination: NUMBER,NAME
Location:
types: NUMBER
Summary claim: "30-year-old" 
Article evidence: "29-year-old" 
Location:
types: NAME
Summary claim: "joining Swans" 
Article evidence: "joining Swansea" 

Example 2 (No Hallucination):
Request: "Generate an appropriate single-sentence summary for the given text such that it includes the main topic of the text. (CNN) -- They're big, strong, and fierce -- and they wear little blue booties. The police dogs in Duesseldorf, Germany are now patrolling the pavement in protective shoes that their police-officer handlers strap onto their paws. The reason? Too many glass shards left by beer drinkers in the city center, said Andre Hartwich, a spokesman for police in Duesseldorf. \"We wondered how can we protect our dogs' feet against glass,\" said Hartwich. \"We looked on the Internet and found these shoes.\" Beer drinkers along the Rhine River and in the city's Altstadt, or Old Town, often discard beer bottles on pebbled walkways. Broken glass poses a problem for the police force's 20 German Shepherds and Belgian Shepherds, Hartwich said. In addition, hooligans and vandals leave behind glass shards around New Year's Eve and during the city's famous Carnival celebrations. So what's a dog to do? Their handlers shelled out 60 euros -- $89 -- for shoes that are also worn by dogs who walk on ice in Alaska. Dogs need a month of training to get used to wearing the shoes, Hartwich said. \"We have to condition the dogs to the shoes,\" he said. E-mail to a friend ."
Summary: "Police dogs in Duesseldorf, Germany are now wearing protective shoes .\nGlass shards left by beer drinkers in the city center are the reason .\nDuesseldorf police force has 20 German and Belgian Shepherds .\nDogs shoes cost €60 ($89) and are also worn by dogs who walk on ice in Alaska ."
Hallucination: None
Location: None

Now evaluate this case:
Request: {article}
Generated Summary: {summary}

Final Judgement and Type can only select one or more of the following answers:NAME,NUMBER,WORD,CONTEXT,OTHER,NOT CHECKABLE,None,
and can only be Responded STRICTLY in the format:
Hallucination: [Final Judgement]
Location:
types: [TYPE]
Summary claim: "[exact text]"
Article evidence: "[relevant text]" 
... (only if Hallucination is not None)
"""

        response = call_api(client, prompt).strip()
        print(response)
        
        hallucination_types = []
        errors = []
        
        # Extract main hallucination types
        type_line = next((line for line in response.split('\n') if line.startswith("Hallucination:")), None)
        if type_line:
            raw_types = type_line.split(":", 1)[1].strip()
            if raw_types != "None":
                hallucination_types = list(set([t.strip() for t in raw_types.split(",")]))
        
        # Error detail parsing
        current_error = None
        for line in response.split('\n'):
            line = line.strip()
            
            # Parse main types
            if line.startswith("Hallucination:"):
                raw_types = line.split(":", 1)[1].strip()
                if raw_types != "None":
                    hallucination_types = [t.strip() for t in raw_types.split(",")]
            
            # Parse error details
            elif line.startswith("types:"):
                if current_error is None:
                    current_error = {"types": [], "claim": None, "evidence": None}
                current_error["types"] = [t.strip() for t in line.split(":", 1)[1].strip().split(",")]
            
            elif line.startswith("Summary claim:"):
                if current_error is None:
                    current_error = {"types": [], "claim": None, "evidence": None}
                current_error["claim"] = line.split(":", 1)[1].strip().strip('"')
            
            elif line.startswith("Article evidence:") and current_error:
                current_error["evidence"] = line.split(":", 1)[1].strip().strip('"')
                errors.append(current_error)
                current_error = None 
        
        # Final type processing
        final_types = "None" if not hallucination_types else ",".join(sorted(hallucination_types))
        return (final_types, errors if errors else [])
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return ("None", [])


def regenerate_summary(client: OpenAI, article: str, wrong_summary: str, errors: List[Dict]) -> Optional[str]:
    try:
        error_details = []
        for error in errors:
            detail = (
                f"Type: {', '.join(error['types'])}\n"
                f"Summary Claim: {error['claim']}\n"
                f"Article Evidence: {error['evidence'] or 'No supporting evidence'}"
            )
            error_details.append(detail)
        prompt = f"""This summary contains inaccuracies. Please regenerate the summary based on the original article, paying special attention to the following errors:
{chr(10).join(error_details)}

Generate a corrected version using ONLY information from the article. Respond with JUST the corrected summary.

Original Article: {article}
Invalid Summary: {wrong_summary}"""

        response = call_api(client, prompt)
        return response.strip()
    except ContentSafetyError:
        return None

def process_entry(client: OpenAI, entry: Dict) -> Optional[Dict]:
    try:
        question = entry["question"]
        input_text = entry["input"]
        reference = entry["answer"][0]
        
        # Generate initial summary
        initial = generate_initial_summary(client, question)
        if initial is None:
            return None
            
        # Detect hallucinations using DeepSeek
        has_hallu, positions = detect_hallucination(client, question, initial)
       
            
        result = {
            "input": input_text,
            "question": question,
            "initial_summary": initial,
            "reference_summary": reference,
            "hallucination_types": "None",  # Default value
            "hallucination_details": None
        }
        
        if has_hallu not in ["None"] and len(positions) > 0:  # Ensure positions is a list and has content
            result.update({
                "hallucination_types": has_hallu,
                "hallucination_details": [
                    {
                        "type": error.get("types", []),
                        "claim": error.get("claim", ""),
                        "evidence": error.get("evidence") or "No supporting evidence"
                    } for error in positions
                ]
            })
        
        # Modify regeneration condition
        if has_hallu not in ["None"] and len(positions) > 0:
            if corrected := regenerate_summary(client, question, initial, positions):
                result["final_summary"] = corrected
            else:
                return None
        else:
            result["final_summary"] = initial

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
    INPUT_JSONL = "data/processed/detector/sum_sampled_50.jsonl"
    OUTPUT_JSONL = "results/Self-Reflection/sum/llama/sum_llama_location_fine-grained_category_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)