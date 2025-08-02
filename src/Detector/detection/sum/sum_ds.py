import json
import re
from typing import Dict, List, Optional
from openai import OpenAI, APIError

class ContentSafetyError(Exception):
    """content safety detection failed"""

def create_client():
    return OpenAI(
        api_key="",
        base_url=""
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
            raise ContentSafetyError("content safety detection failed") from e
        raise
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

def detect_hallucination(client: OpenAI, input_text: str, summary: str, reference_answer: str) -> Dict[str, str]:

    try:
        prompt = f"""
The given task is to evaluate whether the generated summary contains hallucinated content by comparing it with the original article.

You are a linguist. Your task is to strictly compare the summary with the original article, identifying any discrepancies between the two. Focus on factual consistency and logical derivation.
You will be given three inputs:  original Article , Generated Summary (CandidateSummary) to evaluate,and Reference Summary (dataset’s standard factual answer for comparison).

Then, you need to answer the question: Is the generated summary a hallucination or not?
A summary is considered to have hallucination if it contains any additional information that cannot be supported by the original article, or any nonsensical or factually incorrect information.
Example 1 (Hallucination):
original Article: "Rajvir Rai . Lukasz Fabianski has admitted he has already forgotten his time as an Arsenal player. The 29-year-old spent seven years at the Emirates before joining Swansea on a free transfer earlier this summer. Fabianski only started one league game last season - though he did start the FA Cup final victory over Hull in May - and has revealed he has already put his time at Arsenal behind. Hoping to kick on: Lukasz Fabianski has left Arsenal for Swansea in search of more game time . No chance: Lukasz Fabianski is left helpless as Villarreal score in a pre-season friendly . Glory boy: Lukasz Fabianski celebrates winning the FA Cup with Arsenal in May . 'It was not hard leaving Arsenal. I have already kind of forgotten about it. It's behind me,' he told The Sun. 'I had many wonderful years there. But I am a Swansea player now and want to focus on that.'."
Summary: "Keeper spent seven years at Arsenal before joining Swansea in the summer .30-year-old only played one league game last season .\nFabianski did start Arsenal's 3-2 FA Cup final win over Hull in May."
Hallucination:Yes
Reason: Age misstatement (30-year-old vs. original 29-year-old in article)
Example 2 (No Hallucination):
original Article:"(CNN) -- They're big, strong, and fierce -- and they wear little blue booties. The police dogs in Duesseldorf, Germany are now patrolling the pavement in protective shoes that their police-officer handlers strap onto their paws. The reason? Too many glass shards left by beer drinkers in the city center, said Andre Hartwich, a spokesman for police in Duesseldorf. \"We wondered how can we protect our dogs' feet against glass,\" said Hartwich. \"We looked on the Internet and found these shoes.\" Beer drinkers along the Rhine River and in the city's Altstadt, or Old Town, often discard beer bottles on pebbled walkways. Broken glass poses a problem for the police force's 20 German Shepherds and Belgian Shepherds, Hartwich said. In addition, hooligans and vandals leave behind glass shards around New Year's Eve and during the city's famous Carnival celebrations. So what's a dog to do? Their handlers shelled out 60 euros -- $89 -- for shoes that are also worn by dogs who walk on ice in Alaska. Dogs need a month of training to get used to wearing the shoes, Hartwich said. \"We have to condition the dogs to the shoes,\" he said. E-mail to a friend ."
Summary:"Police dogs in Duesseldorf, Germany are now wearing protective shoes .\nGlass shards left by beer drinkers in the city center are the reason .\nDuesseldorf police force has 20 German and Belgian Shepherds .\nDogs shoes cost €60 ($89) and are also worn by dogs who walk on ice in Alaska ."
Hallucination:No
Reason: All claims are factually consistent with original content

Respond STRICTLY in this format:
Hallucination: [yes/no]
Reason: [Your explanation]

Now evaluate this case:
Request: {input_text}
Generated Summary: {summary}
Reference Summary: {reference_answer}"""
        
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

def process_data(input_path: str, output_path: str):
    client = create_client()
    
   
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            
           
            initial_answers = data.get("initial_summary", [])
            if not isinstance(initial_answers, list):
                initial_answers = [initial_answers] if initial_answers else []
            
           
            judgments = []
            for sum in initial_answers:
                result = detect_hallucination(
                    client=client,
                    input_text=data["input"],
                    summary=sum,
                    reference_answer=data["reference_summary"]
                )
                judgments.append(result)
            
           
            output_data = {
                "question": data.get("question", ""),
                "input": data.get("input", ""),
                "initial_summary": data.get("initial_summary", []),
                "hallucination": data.get("hallucination", ""),
                "final_summary": data.get("final_summary", ""),
                "reference_summary": data.get("reference_summary", ""),
                "gpt_final": judgments[0]["gpt_final"] if judgments else "No",
                "gpt_reason": judgments[0]["gpt_reason"] if judgments else "No summary provided"
            }
            
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "~/data/processed/detector/sum_sampled_50.jsonl"
    output_file = "~/results/Detector/sum/sum_ds_50.jsonl"
    
    process_data(input_file, output_file)