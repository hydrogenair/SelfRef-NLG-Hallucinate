#sum Location 300 llama
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

def detect_hallucination(client: OpenAI, article: str, summary: str) -> Tuple[Optional[bool], Optional[List[str]]]:
    try:
        prompt = f"""
The given task is to evaluate whether the generated summary contains hallucinated content by comparing it with the original article.

You are a linguist. Your task is to strictly compare the summary with the original article, identifying any discrepancies between the two. Focus on factual consistency and logical derivation.
You will be given two inputs: Original Request including the original Article and Generated Summary (CandidateSummary) to evaluate.

Then, you need to answer the question: Is the generated summary a hallucination or not? if it's a hallucination, you need to find the location where the hallucination appears.
A summary is considered have hallucination if it contains any additional information that the original article cannot support or any nonsensical or factually incorrect information.

Example 1 (Hallucination):
Request: "Generate an appropriate summary for the given text such that it includes the main topic of the text. By. Rajvir Rai. Lukasz Fabianski has admitted he has already forgotten his time as an Arsenal player. The 29-year-old spent seven years at the Emirates before joining Swansea on a free transfer earlier this summer. Fabianski only started one league game last season - though he did start the FA Cup final victory over Hull in May - and has revealed he has already put his time at Arsenal behind. Hoping to kick on: Lukasz Fabianski has left Arsenal for Swansea in search of more game time. No chance: Lukasz Fabianski is left helpless as Villarreal scores in a pre-season friendly. Glory boy: Lukasz Fabianski celebrates winning the FA Cup with Arsenal in May. 'It was not hard leaving Arsenal. I have already kind of forgotten about it. It's behind me,' he told The Sun. 'I had many wonderful years there. But I am a Swansea player now and want to focus on that.'."
Summary: "Keeper spent seven years at Arsenal before joining Swansea in the summer .30-year-old only played one league game last season .\nFabianski did start Arsenal's 3-2 FA Cup final win over Hull in May."
Hallucination: Yes
Location:
Summary claim: "30-year-old" 
Article evidence: "29-year-old" 
Summary claim: "3-2 score" 
Article evidence: No supporting evidence 

Example 2 (No Hallucination):
Request: "Generate an appropriate single-sentence summary for the given text such that it includes the main topic of the text. (CNN) -- They're big, strong, and fierce -- and they wear little blue booties. The police dogs in Duesseldorf, Germany are now patrolling the pavement in protective shoes that their police-officer handlers strap onto their paws. The reason? Too many glass shards left by beer drinkers in the city center, said Andre Hartwich, a spokesman for police in Duesseldorf. \"We wondered how can we protect our dogs' feet against glass,\" said Hartwich. \"We looked on the Internet and found these shoes.\" Beer drinkers along the Rhine River and in the city's Altstadt, or Old Town, often discard beer bottles on pebbled walkways. Broken glass poses a problem for the police force's 20 German Shepherds and Belgian Shepherds, Hartwich said. In addition, hooligans and vandals leave behind glass shards around New Year's Eve and during the city's famous Carnival celebrations. So what's a dog to do? Their handlers shelled out 60 euros -- $89 -- for shoes that are also worn by dogs who walk on ice in Alaska. Dogs need a month of training to get used to wearing the shoes, Hartwich said. \"We have to condition the dogs to the shoes,\" he said. E-mail to a friend ."
Summary: "Police dogs in Duesseldorf, Germany are now wearing protective shoes .\nGlass shards left by beer drinkers in the city center are the reason .\nDuesseldorf police force has 20 German and Belgian Shepherds .\nDogs shoes cost €60 ($89) and are also worn by dogs who walk on ice in Alaska ."
Hallucination: No
Location: None

Now evaluate this case:
Request: {article}
Generated Summary: {summary}

Final judgment responds STRICTLY in this format:
Hallucination: [Yes/No]
Location:
Summary claim: "[exact text]"
Article evidence: "[relevant text]" 
... (only if Hallucination is Yes)
"""

        response = call_api(client, prompt).strip()
        
        has_hallucination = None
        locations = []
    
        lines = [line.strip() for line in response.split('\n') if line.strip()]
    
        if lines and lines[0].startswith("Hallucination:"):
            hallucination_status = lines[0].split(":", 1)[1].strip().lower()
            has_hallucination = hallucination_status == "yes"
        
            if has_hallucination and len(lines) > 1 and lines[1].startswith("Location:"):
                i = 2
                while i < len(lines):
                    if lines[i].startswith("Summary claim:"):
                        claim = lines[i].split(":", 1)[1].strip().strip('"')
                        i += 1
                        if i < len(lines) and lines[i].startswith("Article evidence:"):
                            ref = lines[i].split(":", 1)[1].strip().strip('"')
                            locations.append({"claim": claim, "reference": ref})
                        i += 1
                    else:
                        i += 1
                    
        return (has_hallucination, locations if has_hallucination else None)
    
    except ContentSafetyError:
        return (None, None)
    except Exception as e:
        print(f"Hallucination detection failed: {str(e)}")
        return (None, None)

def regenerate_summary(client: OpenAI, article: str, wrong_summary: str, positions: List[str]) -> Optional[str]:
    try:
        errors_str = "\n".join(
        f"Discrepancy found:\n"
        f"Summary claim: {loc['claim']}\n"
        f"Article evidence: {loc['reference'] or 'No supporting evidence'}"
        for loc in positions
    )
        prompt = f"""This summary contains inaccuracies. The following discrepancies were found between the summary and the original article:
{errors_str}

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
            
        # Detect hallucinations
        has_hallu, positions = detect_hallucination(client, question, initial)
        if has_hallu is None:
            return None
            
        result = {
            "input": input_text,
            "question": question,
            "initial_summary": initial,
            "hallucination": "yes" if has_hallu else "no",
            "reference_summary": reference,
            "position": positions
        }

        if has_hallu:
            if corrected := regenerate_summary(client, question, initial, positions):
                result["final_summary"] = corrected
                result["hallucination_positions"] = positions
            else:
                return None  # Skip when content safety detection fails
        else:
            result["final_summary"] = initial  # Keep initial summary

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
    OUTPUT_JSONL = "results/Self-Reflection/sum/llama/sum_llama_location_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)