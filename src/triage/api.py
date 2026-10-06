"""REST API so a helpdesk (Zendesk, Freshdesk, Jira Service Management) can call
the triage system from a webhook when a ticket is created.

    uvicorn triage.api:app --reload
    curl -X POST localhost:8000/triage -H 'content-type: application/json' \
         -d '{"id": "T1", "body": "We were charged twice this month"}'
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from . import __version__
from .factory import build_orchestrator
from .models import Ticket, TriageResult

app = FastAPI(title="Support Ticket Triage Agents", version=__version__)


@lru_cache(maxsize=1)
def orchestrator():
    return build_orchestrator()


@app.get("/health")
def health() -> dict:
    orch = orchestrator()
    return {"status": "ok", "backend": type(orch).__name__, "model": getattr(orch, "model", getattr(getattr(orch, "llm", None), "name", None))}


@app.post("/triage", response_model=TriageResult)
def triage(ticket: Ticket) -> TriageResult:
    return orchestrator().triage(ticket)


@app.post("/triage/batch", response_model=list[TriageResult])
def triage_batch(tickets: list[Ticket]) -> list[TriageResult]:
    orch = orchestrator()
    return [orch.triage(t) for t in tickets]
