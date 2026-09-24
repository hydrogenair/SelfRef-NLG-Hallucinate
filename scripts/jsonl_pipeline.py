"""Shared JSONL read/write loop for detection methods."""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

ProcessFn = Callable[[OpenAI, Dict], Optional[Dict]]


def run_jsonl_main(
    client: OpenAI,
    input_file: str,
    output_file: str,
    process_entry: ProcessFn,
) -> None:
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if processed := process_entry(client, entry):
                    outfile.write(json.dumps(processed, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON line skipped")
            except Exception as exc:
                logger.exception("Processing error: %s", exc)
