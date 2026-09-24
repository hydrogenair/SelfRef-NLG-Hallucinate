"""Summarization hallucination evaluation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from openai import OpenAI

from eval.util import (
    rates_to_pair,
    run_llm_judge,
    scan_jsonl_hallucination_rates,
)

TASK = "sum"

JSONL_KEYS = {
    "input_key": "input",
    "initial_key": "initial_summary",
    "final_key": "final_summary",
    "reference_key": "reference_summary",
}

PROMPT = """
The given task is to evaluate whether the generated summary contains hallucinated content by comparing it with the original article.

You are a linguist. Your task is to compare the summary with the original article, focusing on factual consistency. The summary should accurately reflect the information in the original article, but it may use different wording, reasonable generalizations, and natural language paraphrasing. Minor variations in wording or slight generalizations that preserve the core meaning are acceptable and should NOT be considered hallucinations.

Important: The original article may consist of multiple news articles separated by the special token "|||||". In such cases, the generated summary is expected to synthesize information from all parts of the original article. Information drawn from any of the provided articles within the original input is considered factual and consistent with the source, as long as it is accurately represented. Therefore, combining details from different segments into a single coherent summary is acceptable and not a hallucination.

A summary is considered hallucinated if it:

Contains information that directly contradicts the original article.
Introduces specific details, numbers, or claims that are not present in any part of the original article and cannot be logically inferred from it.
Distorts the relationships between entities or events, leading to a different interpretation than the original.
Acceptable variations include:

Using synonyms or alternative phrasing (e.g., "several weeks" for "some three weeks").
Combining related facts into a concise statement (e.g., mentioning multiple details together).
Making reasonable inferences that are strongly implied by the context.

Example 1 (Hallucination):
Article: "By . Rajvir Rai . Lukasz Fabianski has admitted he has already forgotten his time as an Arsenal player. The 29-year-old spent seven years at the Emirates before joining Swansea on a free transfer earlier this summer. Fabianski only started one league game last season - though he did start the FA Cup final victory over Hull in May - and has revealed he has already put his time at Arsenal behind. Hoping to kick on: Lukasz Fabianski has left Arsenal for Swansea in search of more game time . No chance: Lukasz Fabianski is left helpless as Villarreal score in a pre-season friendly . Glory boy: Lukasz Fabianski celebrates winning the FA Cup with Arsenal in May . 'It was not hard leaving Arsenal. I have already kind of forgotten about it. It's behind me,' he told The Sun. 'I had many wonderful years there. But I am a Swansea player now and want to focus on that.'."
Summary: "Keeper spent seven years at Arsenal before joining Swansea in the summer .30-year-old only played one league game last season .\nFabianski did start Arsenal's 3-2 FA Cup final win over Hull in May."
Hallucination:Yes
Reason: Age misstatement (30-year-old vs. original 29-year-old in article)
Example 2 (No Hallucination):
Article:"(CNN) -- They're big, strong, and fierce -- and they wear little blue booties. The police dogs in Duesseldorf, Germany are now patrolling the pavement in protective shoes that their police-officer handlers strap onto their paws. The reason? Too many glass shards left by beer drinkers in the city center, said Andre Hartwich, a spokesman for police in Duesseldorf. \"We wondered how can we protect our dogs' feet against glass,\" said Hartwich. \"We looked on the Internet and found these shoes.\" Beer drinkers along the Rhine River and in the city's Altstadt, or Old Town, often discard beer bottles on pebbled walkways. Broken glass poses a problem for the police force's 20 German Shepherds and Belgian Shepherds, Hartwich said. In addition, hooligans and vandals leave behind glass shards around New Year's Eve and during the city's famous Carnival celebrations. So what's a dog to do? Their handlers shelled out 60 euros -- $89 -- for shoes that are also worn by dogs who walk on ice in Alaska. Dogs need a month of training to get used to wearing the shoes, Hartwich said. \"We have to condition the dogs to the shoes,\" he said. E-mail to a friend ."
Summary:"Police dogs in Duesseldorf, Germany are now wearing protective shoes .\nGlass shards left by beer drinkers in the city center are the reason .\nDuesseldorf police force has 20 German and Belgian Shepherds .\nDogs shoes cost €60 ($89) and are also worn by dogs who walk on ice in Alaska ."
Hallucination:No
Reason: All claims are factually consistent with original content


Now evaluate the following:

Original Article: {source}
Generated Summary: {candidate}

Is the generated summary a hallucination? Respond strictly in the format:
Hallucination: [yes/no]
Reason: [Your explanation]"""


def detect_hallucination(
    client: OpenAI,
    source: str,
    candidate: str,
    reference: str = "",
) -> Dict[str, str]:
    del reference
    prompt = PROMPT.format(source=source, candidate=candidate)
    return run_llm_judge(client, prompt, TASK)


def detect_paraphrase_hallucination(
    client: OpenAI,
    source: str,
    candidate: str,
    reference: str,
) -> Dict[str, str]:
    return detect_hallucination(client, source, candidate, reference)


def evaluate_jsonl(jsonl_path: str, client: OpenAI | None = None) -> Dict[str, Any]:
    return scan_jsonl_hallucination_rates(
        jsonl_path,
        task=TASK,
        detect_fn=detect_hallucination,
        client=client,
        **JSONL_KEYS,
    )


def evaluate_jsonl_pair(jsonl_path: str) -> Tuple[float, float]:
    return rates_to_pair(evaluate_jsonl(jsonl_path))
