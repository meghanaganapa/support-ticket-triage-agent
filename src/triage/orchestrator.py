"""Orchestrator: runs the agents as a pipeline with a human-in-the-loop gate.

    ticket ─► Classifier ─► Priority ─► Knowledge ─► Responder ─► Guardrail ─► review gate
                  │             │                                                 │
                  └── team ◄────┘                                 auto-queue or human review

Each step is timed and recorded in ``trace`` so every decision is auditable:
a support lead can see *why* a ticket was routed and prioritised the way it was.
"""

from __future__ import annotations

import time

from .agents import ClassifierAgent, GuardrailAgent, KnowledgeAgent, PriorityAgent, ResponderAgent
from .data import load_rows, split, to_ticket
from .llm import LLM, get_llm
from .models import ROUTING, SLA_MINUTES, Category, Ticket, TriageResult

CONFIDENCE_THRESHOLD = 0.55
ALWAYS_REVIEW = {Category.refund}  # money leaving the business always gets a human


class TriageOrchestrator:
    def __init__(self, llm: LLM | None = None, training_rows: list[dict] | None = None,
                 known_customers: list[str] | None = None):
        self.llm = llm or get_llm()
        if training_rows is None:
            training_rows, _ = split(load_rows())
        self.classifier = ClassifierAgent(self.llm).fit(
            [to_ticket(r) for r in training_rows], [r["category"] for r in training_rows])
        self.priority = PriorityAgent(self.llm).fit(
            [to_ticket(r) for r in training_rows], [bool(r.get("escalate")) for r in training_rows])
        self.knowledge = KnowledgeAgent()
        self.responder = ResponderAgent(self.llm)
        customers = known_customers or sorted({r.get("customer", "") for r in training_rows} - {""})
        self.guardrail = GuardrailAgent(known_customers=customers)

    def triage(self, ticket: Ticket) -> TriageResult:
        trace: list[dict] = []

        def step(agent: str, fn, *args):
            start = time.perf_counter()
            out = fn(*args)
            trace.append({"agent": agent, "ms": round((time.perf_counter() - start) * 1000, 1)})
            return out

        cls = step("classifier", self.classifier.run, ticket)
        cls.confidence = round(cls.confidence, 2)
        trace[-1].update(category=cls.category.value, confidence=round(cls.confidence, 2))

        pa = step("priority", self.priority.run, ticket, cls.category)
        trace[-1].update(priority=pa.priority.value, escalate=pa.escalate, signals=pa.signals)

        team = ROUTING[cls.category]
        articles = step("knowledge", self.knowledge.run, ticket, cls.category)
        trace[-1].update(articles=[a.id for a in articles])

        draft = step("responder", self.responder.run, ticket, articles, pa, team)
        draft, report = step("guardrail", self.guardrail.run, draft, articles, ticket.customer)
        trace[-1].update(passed=report.passed, issues=report.issues)

        reasons = []
        if pa.escalate:
            reasons.append("escalation signal: " + ", ".join(pa.signals))
        if cls.confidence < CONFIDENCE_THRESHOLD:
            reasons.append(f"low routing confidence ({cls.confidence:.2f})")
        if not report.passed:
            reasons.append("guardrail: " + "; ".join(report.issues))
        if cls.category in ALWAYS_REVIEW:
            reasons.append(f"policy: {cls.category.value} tickets always reviewed")

        return TriageResult(
            ticket_id=ticket.id, category=cls.category, category_confidence=cls.confidence,
            priority=pa.priority, escalate=pa.escalate, team=team,
            sla_minutes=SLA_MINUTES[pa.priority], signals=pa.signals, articles=articles,
            draft_reply=draft, guardrail=report, needs_human_review=bool(reasons),
            review_reasons=reasons, trace=trace,
        )
