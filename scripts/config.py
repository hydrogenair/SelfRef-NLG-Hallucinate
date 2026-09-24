"""Environment-driven configuration for reproducible experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int


def load_llm_settings() -> LLMSettings:
    return LLMSettings(
        api_key=_require("OPENAI_API_KEY"),
        base_url=_require("OPENAI_BASE_URL"),
        model=_optional("LLM_MODEL", "deepseek-chat"),
        temperature=float(_optional("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(_optional("LLM_MAX_TOKENS", "1000")),
    )


@dataclass(frozen=True)
class PathSettings:
    data_dir: str
    result_dir: str
    detect_methods_dir: str
    human_ratings_dir: str | None


def load_path_settings() -> PathSettings:
    human = os.environ.get("HUMAN_RATINGS_DIR")
    return PathSettings(
        data_dir=_optional("DATA_DIR", "data"),
        result_dir=_optional("RESULT_DIR", "results"),
        detect_methods_dir=_optional("DETECT_METHODS_DIR", "methods"),
        human_ratings_dir=human,
    )


def default_dataset_path(task: str, *, n: int = 300) -> str:
    """Return default JSONL path under DATA_DIR for para | sum | d2t."""
    paths = load_path_settings()
    return os.path.join(paths.data_dir, f"{task}_sampled_{n}.jsonl")
