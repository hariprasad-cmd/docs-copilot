"""The only file that talks to a language model.

Two providers, chosen with LLM_PROVIDER in .env:
  - "ollama": free, runs on your own computer (needs the Ollama app running)
  - "claude": Anthropic's Claude API (needs ANTHROPIC_API_KEY)
The rest of the app calls chat() and structured() and doesn't care which one is used.
"""

import os
from functools import lru_cache
from typing import TypeVar

import anthropic
import requests
from pydantic import BaseModel

from app import config

DECLINE_MESSAGE = "Sorry, I can't help with that. Please contact support."

T = TypeVar("T", bound=BaseModel)


class LLMSetupError(RuntimeError):
    """Raised when the model can't be reached because of setup (no key, Ollama not running...)."""


def chat(system: str, user: str) -> str:
    """Send one system prompt + one user message, return the model's text reply."""
    if config.LLM_PROVIDER == "claude":
        response = _claude().beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            output_config={"effort": "low"},  # short factual answers don't need deep thinking
            # If a safety check ever declines the request, the API retries it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            return DECLINE_MESSAGE
        # The response is a list of blocks; keep only the text ones.
        return "".join(block.text for block in response.content if block.type == "text").strip()

    return _ollama([{"role": "system", "content": system}, {"role": "user", "content": user}])


def structured(prompt: str, schema: type[T]) -> T:
    """Ask a question and get the answer back as a validated Pydantic object (used by the eval judge)."""
    if config.LLM_PROVIDER == "claude":
        response = _claude().messages.parse(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        return response.parsed_output

    # Ollama can force its reply to match a JSON schema; Pydantic then checks it.
    text = _ollama([{"role": "user", "content": prompt}], json_schema=schema.model_json_schema())
    return schema.model_validate_json(text)


@lru_cache  # create the client once and reuse it
def _claude() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMSetupError("ANTHROPIC_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=ollama.")
    return anthropic.Anthropic()


def _ollama(messages: list[dict], json_schema: dict | None = None) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},  # same question -> same answer, which makes evals repeatable
    }
    if json_schema:
        payload["format"] = json_schema

    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=300)
    except requests.ConnectionError:
        raise LLMSetupError(f"Could not reach Ollama at {config.OLLAMA_URL}. Open the Ollama app, or run: ollama serve")
    if r.status_code == 404:
        raise LLMSetupError(f"Ollama model '{config.OLLAMA_MODEL}' is not installed. Run: ollama pull {config.OLLAMA_MODEL}")
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
