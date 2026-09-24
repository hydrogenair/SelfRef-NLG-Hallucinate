"""OpenAI-compatible chat client."""

from __future__ import annotations

import logging
from typing import Optional

from openai import APIError, OpenAI

from scripts.config import load_llm_settings

logger = logging.getLogger(__name__)


class ContentSafetyError(Exception):
    """Content safety inspection failed."""


def get_client() -> OpenAI:
    settings = load_llm_settings()
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def chat_completion(client: OpenAI, prompt: str) -> Optional[str]:
    settings = load_llm_settings()
    try:
        completion = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        return completion.choices[0].message.content
    except APIError as exc:
        if exc.code == 400 and getattr(exc, "type", "") == "data_inspection_failed":
            raise ContentSafetyError("Content safety check failed") from exc
        raise
    except Exception as exc:
        raise RuntimeError(f"API call failed: {exc}") from exc
