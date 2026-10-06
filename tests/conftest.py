from __future__ import annotations

import os
from dataclasses import dataclass, field

# Tests never call a paid API, even if a developer's .env selects one.
os.environ["TRIAGE_LLM"] = "offline"

import pytest

from triage.llm import OfflineLLM
from triage.orchestrator import TriageOrchestrator


@dataclass
class FakeLLM:
    """Returns scripted JSON so LLM code paths are tested without network or cost."""

    replies: list = field(default_factory=list)
    name: str = "fake"
    calls: list = field(default_factory=list)

    def complete_json(self, system: str, user: str):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else None


@pytest.fixture(scope="session")
def orch() -> TriageOrchestrator:
    return TriageOrchestrator(llm=OfflineLLM())
