# SelfRef‑NLG‑Hallucinate

> Code for paper **Self‑Reflection Often Fails to Address Hallucination in Classical Natural Language Generation Tasks: An Empirical Study**

This repository contains the code for our empirical investigation of **self‑reflection for hallucination mitigation** on three classical NLG tasks: **summarisation, paraphrasing, and data‑to‑text generation**\.

We evaluate **eight hallucination‑detection strategies** spanning coarse‑binary judgment, category annotation, error localisation, scoring, and free‑form feedback, across multiple open‑source and frontier LLMs\. Our experiments show that self‑reflection only yields a limited, inconsistent reduction of hallucination; the primary bottleneck is models' poor self‑diagnosis ability\. Iterative self‑reflection brings negligible further gains\. Using a stronger external annotator can improve factual consistency but weakens the lightweight appeal of self‑reflection\.

> 📄 Paper: Self‑Reflection Often Fails to Address Hallucination in Classical Natural Language Generation Tasks: An Empirical Study
> 🔗 Code repository: [https://github\.com/xxx/SelfRef](https://github.com/xxx/SelfRef)‑NLG‑Hallucinate

## Directory layout

```Plain Text
publication/
├── HEDS.pdf                # Full Human Evaluation Datasheet (see below)
├── data/
├── eval/
│   ├── para.py / sum.py / d2t.py
│   ├── util.py
│   ├── hallucination.py
│   ├── judge.py              # CLI: hallucination rates on a JSONL
│   └── fleiss_kappa.py
├── methods/
├── scripts/                  # config, LLM client, JSONL pipeline, run_batch.py
└── analysis/
```

## Human Evaluation Datasheet

`HEDS.pdf` follows the template described in [HEDS 3.0 (Belz and Thomson ,2025)](https://aclanthology.org/2025.gem-1.6/).It archives the complete human‑annotation materials used for selecting hallucination detectors, covering annotator background, full annotation instructions, labelling definitions, and quality‑control protocols\. It computes inter‑annotator agreement \(Fleiss’ Kappa\) via `eval/fleiss_kappa.py` .

## Hallucination detector selection

1. **Samples:** `data/*_sampled_50.jsonl`\.

2. **Human protocol:** `HEDS.pdf` \(annotators and instructions\)\.

3. **Human agreement \(optional\):** `eval/fleiss_kappa.py` \.

4. **LLM evaluation:** same task modules as Part 2\.

```bash
python eval/judge.py --task para \
  --input-jsonl path/to/outputs.jsonl --output-json rates.json
```

## Hallucination detection pipeline

1. **Data:** `data/{task}_sampled_300.jsonl`\.

2. **Methods:** `methods/*.py`\.

3. **Batch:** `scripts/run_batch.py`\.

4. **Power analysis:** `analysis/power_analysis.py`\.

```bash
export TASK=d2t RESULT_DIR=results/d2t_run1 RUNS=5
python scripts/run_batch.py
python analysis/power_analysis.py
```

## `scripts/` 

|Module|Role|
|---|---|
|`config.py`|Environment variables \(API, paths\)|
|`llm_client.py`|OpenAI‑compatible chat calls|
|`jsonl_pipeline.py`|JSONL loop used by `methods/`|
|`logging_config.py`|Logging setup|
|`run_batch.py`|Run all methods \+ eval rates|

## License

\[Apache‑2\.0\]

