"""Pluggable LLM backends.

Every agent talks to an ``LLM`` with one method, ``complete_json``. That keeps
agents independent of any vendor and lets the whole system run three ways:

* ``offline``  - no LLM at all. Agents fall back to an ML model and rules, so
                 tests, CI and the demo run with zero cost and zero API keys.
* ``anthropic`` - Claude via the Anthropic API (set ANTHROPIC_API_KEY).
* ``openai``    - any OpenAI-compatible endpoint: OpenAI, Azure OpenAI, or a
                 free local model through Ollama (set OPENAI_BASE_URL).

Select with the TRIAGE_LLM environment variable.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol


class LLM(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> dict | None:
        """Return the model's reply parsed as JSON, or None if unavailable."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "seconds": round(self.seconds, 2)}


def _parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (models sometimes add prose)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class OfflineLLM:
    """No-op backend. Agents detect ``None`` and use their deterministic fallback."""

    name: str = "offline"
    usage: Usage = field(default_factory=Usage)

    def complete_json(self, system: str, user: str) -> dict | None:
        return None


@dataclass
class AnthropicLLM:
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    name: str = "anthropic"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        import anthropic  # imported lazily so offline mode needs no SDK

        self._client = anthropic.Anthropic()

    def complete_json(self, system: str, user: str) -> dict | None:
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0,
            system=system + "\nRespond with a single JSON object only.",
            messages=[{"role": "user", "content": user}],
        )
        self.usage.calls += 1
        self.usage.input_tokens += msg.usage.input_tokens
        self.usage.output_tokens += msg.usage.output_tokens
        self.usage.seconds += time.perf_counter() - start
        text = "".join(block.text for block in msg.content if block.type == "text")
        return _parse_json(text)


@dataclass
class OpenAICompatLLM:
    """OpenAI, Azure OpenAI or Ollama (e.g. OPENAI_BASE_URL=http://localhost:11434/v1)."""

    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "llama3.1"))
    name: str = "openai"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        )

    def complete_json(self, system: str, user: str) -> dict | None:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        self.usage.calls += 1
        if resp.usage:
            self.usage.input_tokens += resp.usage.prompt_tokens
            self.usage.output_tokens += resp.usage.completion_tokens
        self.usage.seconds += time.perf_counter() - start
        return _parse_json(resp.choices[0].message.content or "")


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Never overrides variables already set."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if value.strip():
                    os.environ.setdefault(key.strip(), value.strip())


def get_llm(kind: str | None = None) -> LLM:
    load_dotenv()
    kind = (kind or os.getenv("TRIAGE_LLM", "offline")).lower()
    if kind == "anthropic":
        return AnthropicLLM()
    if kind in ("openai", "ollama", "azure"):
        return OpenAICompatLLM()
    return OfflineLLM()
