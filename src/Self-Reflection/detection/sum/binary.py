#sum binary 300 llama
import json
import re
import os
from typing import Dict, Optional
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
        match = re.search(r'Summary:\s*(.+)$', response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()
    except ContentSafetyError:
        print(f"Content safety blocked: first 20 characters of article {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, article: str, summary: str) -> Optional[bool]:
    try:
        prompt = f"""
The given task is to evaluate whether the generated summary contains hallucinated content by comparing it with the original article.

You are a linguist. Your task is to strictly compare the summary with the original article, identifying any discrepancies between the two. Focus on factual consistency and logical derivation.
You will be given two inputs: Original Request including original Article and Generated Summary (CandidateSummary) to evaluate.

Then, you need to answer the question: Is the generated summary a hallucination or not?
A summary is considered to have hallucination if it contains any additional information that cannot be supported by the original article, or any nonsensical or factually incorrect information.
Example 1 (Hallucination):
Request: "Generate an appropriate summary for the given text such that it includes the main topic of the text.By . Rajvir Rai . Lukasz Fabianski has admitted he has already forgotten his time as an Arsenal player. The 29-year-old spent seven years at the Emirates before joining Swansea on a free transfer earlier this summer. Fabianski only started one league game last season - though he did start the FA Cup final victory over Hull in May - and has revealed he has already put his time at Arsenal behind. Hoping to kick on: Lukasz Fabianski has left Arsenal for Swansea in search of more game time . No chance: Lukasz Fabianski is left helpless as Villarreal score in a pre-season friendly . Glory boy: Lukasz Fabianski celebrates winning the FA Cup with Arsenal in May . 'It was not hard leaving Arsenal. I have already kind of forgotten about it. It's behind me,' he told The Sun. 'I had many wonderful years there. But I am a Swansea player now and want to focus on that.'."
Summary: "Keeper spent seven years at Arsenal before joining Swansea in the summer .30-year-old only played one league game last season .\nFabianski did start Arsenal's 3-2 FA Cup final win over Hull in May."
Example 2 (No Hallucination):
Request:"Generate an appropriate single-sentence summary for the given text such that it includes the main topic of the text.(CNN) -- They're big, strong, and fierce -- and they wear little blue booties. The police dogs in Duesseldorf, Germany are now patrolling the pavement in protective shoes that their police-officer handlers strap onto their paws. The reason? Too many glass shards left by beer drinkers in the city center, said Andre Hartwich, a spokesman for police in Duesseldorf. \"We wondered how can we protect our dogs' feet against glass,\" said Hartwich. \"We looked on the Internet and found these shoes.\" Beer drinkers along the Rhine River and in the city's Altstadt, or Old Town, often discard beer bottles on pebbled walkways. Broken glass poses a problem for the police force's 20 German Shepherds and Belgian Shepherds, Hartwich said. In addition, hooligans and vandals leave behind glass shards around New Year's Eve and during the city's famous Carnival celebrations. So what's a dog to do? Their handlers shelled out 60 euros -- $89 -- for shoes that are also worn by dogs who walk on ice in Alaska. Dogs need a month of training to get used to wearing the shoes, Hartwich said. \"We have to condition the dogs to the shoes,\" he said. E-mail to a friend ."
Summary:"Police dogs in Duesseldorf, Germany are now wearing protective shoes .\nGlass shards left by beer drinkers in the city center are the reason .\nDuesseldorf police force has 20 German and Belgian Shepherds .\nDogs shoes cost €60 ($89) and are also worn by dogs who walk on ice in Alaska ."

Now evaluate this case:
Request: {article}
Generated Summary: {summary}
Final judgment responds STRICTLY in this format: Hallucination: [Yes/No]
"""

        response = call_api(client, prompt).strip()
        return response.lower().startswith("hallucination: yes")
    except ContentSafetyError:
        return None

def regenerate_summary(client: OpenAI, article: str, wrong_summary: str) -> Optional[str]:
    try:
        prompt = f"""This summary contains inaccuracies. Generate a corrected version using ONLY information from the article. Respond with JUST the corrected summary.

Request: {article}
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
        has_hallu = detect_hallucination(client, question, initial)
        if has_hallu is None:
            return None
            
        result = {
            "input": input_text,
            "question": question,
            "initial_summary": initial,
            "hallucination": "yes" if has_hallu else "no",
            "reference_summary": reference
        }

        if has_hallu:
            if corrected := regenerate_summary(client, question, initial):
                result["final_summary"] = corrected
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
    OUTPUT_JSONL = "results/Self-Reflection/sum/llama/sum_llama_binary_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)