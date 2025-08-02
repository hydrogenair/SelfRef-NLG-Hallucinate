#sum fine-grained category 300 llama 
import re
import json
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
        return response.strip()
    except ContentSafetyError:
        print(f"Content safety blocked: first 20 characters of article {question[:20]}...")
        return None

def detect_hallucination(client: OpenAI, article: str, summary: str) -> Optional[str]:
    try:
        prompt = f"""
The given task is to evaluate whether the generated summary contains hallucinated content by comparing it with the original article and classify the type of hallucination if present.

You are a senior editor specializing in factual verification. 
You will be given two inputs: Original Request including the original Article and Generated Summary to evaluate.
Your task is to strictly compare the summary with the original article, identifying discrepancies between the two. 

Then, you need to answer the question: Is the generated summary a hallucination or not?
A summary is considered a hallucination if it contains any additional information that the original article cannot support or any nonsensical or factually incorrect information.
If it's a hallucination, you need to categorize the hallucinaition into one of the following categories : 
1.  Incorrect named entity(NAME): This includes people, places, organizations, and days of the week.
2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)
If the summary is fully consistent and all information is supported by the article, the hallucination type is None.
For each claim in the summary, determine if it falls into any of the above categories. After evaluating all claims, provide the final judgment by listing all applicable hallucination types separated by commas if multiple. If there are none, state None.

Example 1 (Contradictory Hallucination):
Request: "Generate an appropriate summary for the given text such that it includes the main topic of the text. By. Rajvir Rai. Lukasz Fabianski has admitted he has already forgotten his time as an Arsenal player. The 29-year-old spent seven years at the Emirates before joining Swansea on a free transfer earlier this summer. Fabianski only started one league game last season - though he did start the FA Cup final victory over Hull in May - and has revealed he has already put his time at Arsenal behind. Hoping to kick on: Lukasz Fabianski has left Arsenal for Swansea in search of more game time. No chance: Lukasz Fabianski is left helpless as Villarreal scores in a pre-season friendly. Glory boy: Lukasz Fabianski celebrates winning the FA Cup with Arsenal in May. 'It was not hard leaving Arsenal. I have already kind of forgotten about it. It's behind me,' he told The Sun. 'I had many wonderful years there. But I am a Swansea player now and want to focus on that.'."
Generated Summary: "Keeper spent seven years at Arsenal before joining Swansea in the summer .30-year-old only played one league game last season .\nFabianski did start Arsenal's 3-2 FA Cup final win over Hull in May."
Final judgment: NUMBER

Example 2 (No Hallucination):
Request: "Generate an appropriate single-sentence summary for the given text such that it includes the main topic of the text. (CNN) -- They're big, strong, and fierce -- and they wear little blue booties. The police dogs in Duesseldorf, Germany are now patrolling the pavement in protective shoes that their police-officer handlers strap onto their paws. The reason? Too many glass shards left by beer drinkers in the city center, said Andre Hartwich, a spokesman for police in Duesseldorf. \"We wondered how can we protect our dogs' feet against the glass,\" said Hartwich. \"We looked on the Internet and found these shoes.\" Beer drinkers along the Rhine River and in the city's Altstadt, or Old Town, often discard beer bottles on pebbled walkways. Broken glass poses a problem for the police force's 20 German Shepherds and Belgian Shepherds, Hartwich said. In addition, hooligans and vandals leave behind glass shards around New Year's Eve and during the city's famous Carnival celebrations. So what's a dog to do? Their handlers shelled out 60 euros -- $89 -- for shoes that are also worn by dogs who walk on ice in Alaska. Dogs need a month of training to get used to wearing the shoes, Hartwich said. \"We have to condition the dogs to the shoes,\" he said. E-mail to a friend ."
Generated Summary: "Police dogs in Duesseldorf, Germany are now wearing protective shoes .\nGlass shards left by beer drinkers in the city center are the reason .\nDuesseldorf police force has 20 German and Belgian Shepherds .\nDogs shoes cost €60 ($89) and are also worn by dogs who walk on ice in Alaska ."
Final judgment: None

Now evaluate this case:
Request: {article}
Generated Summary: {summary}

Final Judgement can only select one or more of the following answers:NAME,NUMBER,WORD,CONTEXT,OTHER,NOT CHECKABLE,None,
and can only be Responded STRICTLY in the format:
Hallucination:[your Final Judgement]
"""

        response = call_api(client, prompt).strip()
        match = re.search(r'Hallucination:\s*(.+)', response)
        if not match:
            return None  # Format error treated as detection failure
        types_str = match.group(1).strip()
        return types_str if types_str != 'None' else 'None'
    except ContentSafetyError:
        return None
    except Exception as e:
        print(f"Error detecting hallucination: {str(e)}")
        return None

def regenerate_summary(client: OpenAI, article: str, wrong_summary: str, hallucination_types: str) -> Optional[str]:
    try:
        prompt = f"""
        Hallucinations are categorized as follows:
        1.  Incorrect named entity(NAME): This includes people, places, organizations, and days of the week.
        2.  Incorrect number (NUMBER): This includes numbers that are spelled out as well as digits.
        3.  Incorrect word (WORD): A word or phrase that is not one of the above and is incorrect. 
        4.  Context error (CONTEXT): A word or phrase that causes an incorrect inference because of context or discourse. 
        5.  Not checkable (NOT CHECKABLE): A statement that cannot be checked; either the information is not available or it is too time-consuming to check.
        6.  Other (OTHER): Any other type of mistake (such as nonsensical phrases)
        This summary contains {hallucination_types} hallucinations. Generate a corrected version using ONLY information from the article. Respond with JUST the corrected summary.
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
        hallucination_types = detect_hallucination(client, question, initial)
        if hallucination_types == "NONE":
            hallucination_types='None'
        if hallucination_types is None:
            return None
        has_hallu = hallucination_types != 'None'
            
        result = {
            "input": input_text,
            "question": question,
            "initial_summary": initial,
            "hallucination": "yes" if has_hallu else "no",
            "hallucination_types": hallucination_types,
            "reference_summary": reference
        }

        if has_hallu:
            types = [t.strip() for t in hallucination_types.split(',')]
            relevant_types = [t for t in types if t in ['NAME','NUMBER','WORD','CONTEXT','OTHER','NOT CHECKABLE']]
            if relevant_types:
                if corrected := regenerate_summary(client, question, initial, ', '.join(relevant_types)):
                    result["final_summary"] = corrected
                else:
                    return None
            else:
                result["final_summary"] = initial
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
    OUTPUT_JSONL = "results/Self-Reflection/sum/llama/sum_llama_fine-grained_category_50.jsonl"
    main(INPUT_JSONL, OUTPUT_JSONL)
    