## Overview

​	Hallucination remains a critical challenge in the era of Large Language Models (LLMs). Recent studies have proposed self-reflection as a promising strategy to mitigate hallucinations, based on the assumption that models can detect and correct their errors. While this approach has shown remarkable success in question-answering systems, its effectiveness in classic Natural Language Generation (NLG) tasks remains unclear. In this study, we conduct an empirical investigation into the effectiveness of self-reflection across three representative NLG tasks: summarisation, paraphrasing, and data-to-text generation. We explore a broad spectrum of strategies for instructing LLMs to detect their hallucinations, ranging from coarse-grained to fine-grained detection methods. Surprisingly, regardless of the granularity of the detection approach, self-reflection consistently performs poorly in mitigating hallucinations across all three NLG tasks. A closer examination reveals that LLMs demonstrate a limited ability to accurately identify their own mistakes during self-reflection. As a result, they often retrieve insufficient or incorrect evidence for subsequent corrections. Based on this observation, we also find that reflection can be effective only when a significantly stronger LLM is used to assist a weaker LLM in detecting hallucinations—a solution that is arguably impractical in real-world applications.

​	This project focuses on addressing hallucination issues in three key natural language generation (NLG) tasks: **Data to Text (D2T)**, **Paraphrasing**, and **Summarization**. It provides a framework to generate initial outputs, detect hallucinated content (inaccuracies or unsupported information), and regenerate corrected versions using state-of-the-art language models.

## Supported Tasks

The project targets three core NLG tasks, as defined in `data/selected_tasks.txt`:

1. **Data to Text (D2T)**: Converting structured data into coherent natural language descriptions.
2. **Paraphrasing**: Rewriting text while preserving its original meaning.
3. **Summarization**: Generating concise summaries of longer texts.

## Directory Structure

```plaintext
Self-Reflection-Failed/
├── data/
│   ├── selected_tasks.txt       # List of supported NLG tasks
│   └── processed/               # Input data files (JSONL format) for each task
├── src/
│   ├── Explanation/             # Modules with hallucination detection explanations
│   │   └── detection/
│   │       ├── para/            # Paraphrasing task
│   │       └── d2t/             # D2T task
│   ├── Self-Reflection/         # Self-reflection-based correction modules
│   │   └── detection/
│   │       ├── para/            # Paraphrasing task
│   │       ├── d2t/             # D2T task (with submodules for different models)
│   │       └── sum/             # Summarization task
│   ├── External Annotation/     # External annotation-based correction
│   │   └── detection/para/
│   └── Detector/                # Core hallucination detection utilities
│       └── detection/para/
└── results/                     # Output files (JSONL format) with generated/corrected texts
```

## Key Features

### 1. Text Generation

- Generates initial outputs for D2T, paraphrasing, and summarization tasks using models like `meta/llama-3.1-8b-instruct`, `qwen2.5` series, and `deepseek-chat`.
- Handles content safety checks to filter inappropriate content (via `ContentSafetyError`).

### 2. Hallucination Detection

- Adopt a series of strategies to guide LLM in detecting illusions, ranging from coarse-grained to fine-grained detection methods
- In External Annotation, DeepSeeker V3 is deployed as a powerful annotator to detect illusions in weaker LLM raw reactions
- Explanation is the impact of adding explanatory reasoning during the external annotation process.

### 3. Text Regeneration

- Automatically regenerates outputs to fix detected hallucinations.
- Uses error-specific feedback (e.g., "incorrect member count") to guide correction.
- Ensures corrected outputs adhere strictly to source data.

## Dependencies

- Python 3.8+
- `openai` library (for API interactions with language models)
- JSONL file support (for input/output data)

Install dependencies with:

```bash
pip install openai
```

## Usage

### 1. Configuration

- Set up API credentials for your language models:
  - Update `base_url` and `api_key` in `create_client()`, `create_qwen_client()`, and `create_deepseek_client()` functions (e.g., in `src/Self-Reflection/detection/para/binary.py`).

### 2. Input Data

- Prepare input data in JSONL format, where each line contains a JSON object with:
  - `question`: The generation task/query.
  - `input`: Source text/data for the task.
  - `answer`: Reference answer(s) for evaluation.

### 3. Run Tasks

Execute the main script for your target task and model. Example for paraphrasing with binary hallucination detection:

```bash
python -u src/Self-Reflection/detection/para/binary.py
```

### 4. Output

- Results are saved in JSONL format (e.g., `results/Self-Reflection/para/llama/para_llama_binary_300.jsonl`).
- Each output entry includes:
  - `input`: Source input.
  - `question`: Task query.
  - `initial_answer`: First generated output.
  - `hallucination`: "yes" or "no" (detection result).
  - `final_answer`: Corrected output (if hallucination was detected).
  - `reference_answer`: Ground truth reference.
  - `reason`: Explanation for hallucination (if applicable).

## Notes

- **Content Safety**: The framework includes `ContentSafetyError` to handle cases where generated content fails safety checks (marked as `[Blocked]`).
- **Model Compatibility**: Supports multiple models (Llama, Qwen, DeepSeek) – adjust `model` parameter in `call_api()` functions to switch models.
- **Extensibility**: Add new tasks by extending the detection/regeneration logic (follow the structure of existing task modules).