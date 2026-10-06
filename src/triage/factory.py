"""Pick the orchestrator from TRIAGE_LLM (or an explicit backend name).

    offline | anthropic | openai  -> TriageOrchestrator (fixed pipeline, one LLM call per agent)
    agent-sdk                     -> AgentSDKOrchestrator (Claude plans and calls the agents as tools)
"""

from __future__ import annotations

import os

from .llm import get_llm, load_dotenv


def build_orchestrator(backend: str | None = None, training_rows: list[dict] | None = None):
    load_dotenv()
    backend = (backend or os.getenv("TRIAGE_LLM", "offline")).lower()
    if backend in ("agent-sdk", "agent_sdk", "claude-agent"):
        from .agent_sdk import AgentSDKOrchestrator

        return AgentSDKOrchestrator(training_rows=training_rows)
    from .orchestrator import TriageOrchestrator

    return TriageOrchestrator(llm=get_llm(backend), training_rows=training_rows)
