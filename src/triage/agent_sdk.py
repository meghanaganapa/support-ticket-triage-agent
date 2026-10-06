"""Claude Agent SDK orchestrator: Claude plans the triage and calls the agents as tools.

    ticket ─► Claude (orchestrator) ──calls──► classify_ticket      (ML classifier agent)
                                     ├─calls──► assess_priority      (rules + learned models)
                                     ├─calls──► search_help_centre   (knowledge agent)
                                     ├─calls──► check_reply          (guardrail agent)
                                     └─calls──► submit_triage        (final structured decision)
                  then, in code (Claude cannot skip this):
                    priority floor · guardrails re-run · review gate · cost + trace

Claude adds judgement the tools lack: reading unfamiliar wording, overriding a
low-confidence route, and writing the reply. The code keeps the guarantees:
Claude may RAISE priority but never lower what the rules detected, the final
reply is re-checked by the deterministic guardrail, and refunds, escalations
and low confidence still go to a human.

Requires the ``claude-agent-sdk`` package and ANTHROPIC_API_KEY. If either is
missing, or a run fails, the ticket falls back to the offline orchestrator and
the trace says so, so the service never goes down with the LLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from typing import Any

from .agents.priority import ORDER
from .llm import OfflineLLM, load_dotenv
from .models import (
    ROUTING,
    SLA_MINUTES,
    Category,
    DraftReply,
    Priority,
    Ticket,
    TriageResult,
)
from .orchestrator import ALWAYS_REVIEW, TriageOrchestrator

SERVER = "triage"
TOOL_NAMES = ["classify_ticket", "assess_priority", "search_help_centre", "check_reply", "submit_triage"]
ALLOWED_TOOLS = [f"mcp__{SERVER}__{name}" for name in TOOL_NAMES]

SYSTEM_PROMPT = """You are the triage lead for CloudLedger customer support (an invoicing app for small
businesses). Triage the ticket you are given using ONLY your tools, in this order:

1. classify_ticket - the ML router's team suggestion and confidence.
2. assess_priority - rule and model signals for urgency (P1 outage/security/data exposure,
   P2 money or access at risk or legal threat, P3 normal, P4 feature idea).
3. search_help_centre - find the article(s) that answer the customer.
4. Write a reply: friendly plain Australian English, under 120 words, using ONLY facts from the
   articles, citing them like [KB-102]. Never promise refunds, credits or timelines the articles
   don't state. If escalated, say a specialist is on it.
5. check_reply - fix and re-check until it passes (at most twice).
6. submit_triage - your final decision. You may overrule the ML category when the ticket
   clearly belongs elsewhere. Escalate outages, security incidents, data exposure and legal or
   regulator threats even if the tools missed them. Give an honest confidence.

Do not answer in plain text; finish by calling submit_triage."""


def _text(payload: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _higher(a: Priority, b: Priority) -> Priority:
    return a if ORDER.index(a) <= ORDER.index(b) else b


class TicketTools:
    """The agents exposed as tools for ONE ticket. Plain async functions, so they
    are unit-testable without Claude; ``sdk_server()`` wraps them for the SDK."""

    def __init__(self, base: TriageOrchestrator, ticket: Ticket):
        self.base, self.ticket = base, ticket
        self.calls: list[dict] = []
        self.submitted: dict | None = None
        self.articles: dict = {}

    def _log(self, name: str, args: dict, out: Any) -> dict:
        self.calls.append({"tool": name, "args": args})
        return _text(out)

    async def classify_ticket(self, args: dict) -> dict:
        c = self.base.classifier._ml_predict(self.ticket)
        return self._log("classify_ticket", args, {
            "category": c.category.value, "confidence": round(c.confidence, 2),
            "team": ROUTING[c.category], "teams": {k.value: v for k, v in ROUTING.items()}})

    async def assess_priority(self, args: dict) -> dict:
        category = Category(args.get("category", "technical"))
        pa = self.base.priority.assess(self.ticket, category)
        return self._log("assess_priority", args, {
            "priority": pa.priority.value, "escalate": pa.escalate, "signals": pa.signals,
            "sla_minutes": SLA_MINUTES[pa.priority]})

    async def search_help_centre(self, args: dict) -> dict:
        category = Category(args["category"]) if args.get("category") in Category._value2member_map_ else None
        found = self.base.knowledge.search(args.get("query") or self.ticket.body, category, k=3)
        for a in found:
            self.articles[a.id] = a
        return self._log("search_help_centre", args,
                         [{"id": a.id, "title": a.title, "text": a.text, "score": a.score} for a in found])

    async def check_reply(self, args: dict) -> dict:
        draft = DraftReply(text=args.get("text", ""), cited_articles=list(args.get("cited_articles", [])))
        clean, report = self.base.guardrail.run(draft, list(self.articles.values()), self.ticket.customer)
        return self._log("check_reply", args, {"passed": report.passed, "issues": report.issues,
                                               "redacted_text": clean.text})

    async def submit_triage(self, args: dict) -> dict:
        self.submitted = args
        return self._log("submit_triage", args, {"status": "received"})

    def sdk_server(self):
        from claude_agent_sdk import create_sdk_mcp_server, tool

        cats = [c.value for c in Category]
        specs = [
            ("classify_ticket", "ML router: suggested category, team and confidence for this ticket.",
             {"type": "object", "properties": {}}, self.classify_ticket),
            ("assess_priority", "Rule + learned-model urgency signals for this ticket in a category.",
             {"type": "object", "properties": {"category": {"type": "string", "enum": cats}},
              "required": ["category"]}, self.assess_priority),
            ("search_help_centre", "Search CloudLedger help-centre articles.",
             {"type": "object", "properties": {"query": {"type": "string"},
                                               "category": {"type": "string", "enum": cats}},
              "required": ["query"]}, self.search_help_centre),
            ("check_reply", "Guardrail check of a draft reply: PII, unauthorised promises, citations.",
             {"type": "object", "properties": {"text": {"type": "string"},
                                               "cited_articles": {"type": "array", "items": {"type": "string"}}},
              "required": ["text"]}, self.check_reply),
            ("submit_triage", "Submit the final triage decision. Call exactly once, last.",
             {"type": "object", "properties": {
                 "category": {"type": "string", "enum": cats},
                 "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                 "escalate": {"type": "boolean"},
                 "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                 "rationale": {"type": "string"},
                 "reply": {"type": "string"},
                 "cited_articles": {"type": "array", "items": {"type": "string"}}},
              "required": ["category", "priority", "escalate", "confidence", "rationale", "reply"]},
             self.submit_triage),
        ]
        return create_sdk_mcp_server(SERVER, tools=[tool(n, d, s)(fn) for n, d, s, fn in specs])


QueryFn = Callable[..., Any]


class AgentSDKOrchestrator:
    """Same ``triage(ticket) -> TriageResult`` interface as the offline orchestrator."""

    def __init__(self, training_rows: list[dict] | None = None, model: str | None = None,
                 max_budget_usd: float = 0.10, max_turns: int = 14, query_fn: QueryFn | None = None):
        load_dotenv()
        self.base = TriageOrchestrator(llm=OfflineLLM(), training_rows=training_rows)
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self.max_budget_usd, self.max_turns = max_budget_usd, max_turns
        self._query = query_fn
        self.cost_usd, self.runs, self.fallbacks = 0.0, 0, 0
        self.input_tokens = self.output_tokens = 0

    def usage_summary(self) -> dict:
        return {"cost_usd": round(self.cost_usd, 5), "runs": self.runs, "fallbacks": self.fallbacks,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    def _options(self, tools: TicketTools):
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            tools=[],                      # no built-in file/shell/web tools: only the triage tools
            mcp_servers={SERVER: tools.sdk_server()},
            allowed_tools=ALLOWED_TOOLS,
            setting_sources=[],            # ignore any local Claude Code settings
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
        )

    async def _run(self, ticket: Ticket, tools: TicketTools) -> Any:
        query = self._query
        if query is None:
            from claude_agent_sdk import query
        prompt = f"Ticket {ticket.id} from {ticket.customer} ({ticket.plan} plan)\nSubject: {ticket.subject}\n\n{ticket.body}"
        result = None
        async for message in query(prompt=prompt, options=self._options(tools)):
            if type(message).__name__ == "ResultMessage":
                result = message
        return result

    def triage(self, ticket: Ticket) -> TriageResult:
        tools = TicketTools(self.base, ticket)
        self.current_tools = tools  # exposed so tests can script a fake Claude against real tools
        start = time.perf_counter()
        error = None
        try:
            if self._query is None and not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            result = asyncio.run(self._run(ticket, tools))
        except Exception as exc:  # noqa: BLE001 - SDK/CLI missing, network, budget: never take the service down
            result, error = None, f"{type(exc).__name__}: {exc}"
        self.runs += 1
        if result is not None:
            self.cost_usd += float(getattr(result, "total_cost_usd", 0) or 0)
            usage = getattr(result, "usage", None) or {}
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)

        if tools.submitted is None:
            self.fallbacks += 1
            fallback = self.base.triage(ticket)
            fallback.trace.insert(0, {"agent": "claude-orchestrator", "fallback": True,
                                      "reason": error or "no submit_triage call"})
            return fallback
        return self._finalise(ticket, tools, start, result)

    def _finalise(self, ticket: Ticket, tools: TicketTools, start: float, result: Any) -> TriageResult:
        s = tools.submitted or {}
        try:
            category = Category(s["category"])
            claude_priority = Priority(s["priority"])
        except (KeyError, ValueError):
            return self.base.triage(ticket)
        confidence = round(max(0.0, min(1.0, float(s.get("confidence", 0.5)))), 2)

        # Guarantees enforced in code, whatever Claude decided:
        floor = self.base.priority.assess(ticket, category)                 # 1. priority floor
        priority = _higher(floor.priority, claude_priority)
        escalate = bool(s.get("escalate")) or floor.escalate
        articles = list(tools.articles.values()) or self.base.knowledge.run(ticket, category)
        draft = DraftReply(text=str(s.get("reply", "")), cited_articles=list(s.get("cited_articles", [])))
        draft, report = self.base.guardrail.run(draft, articles, ticket.customer)  # 2. guardrails re-run

        reasons = []                                                        # 3. review gate
        if escalate:
            reasons.append("escalation: " + (", ".join(floor.signals) or "claude judgement"))
        if confidence < self.base.thresholds["confidence"]:
            reasons.append(f"low confidence ({confidence:.2f})")
        if not report.passed:
            reasons.append("guardrail: " + "; ".join(report.issues))
        if category in ALWAYS_REVIEW:
            reasons.append(f"policy: {category.value} tickets always reviewed")

        trace = [{"agent": "claude-orchestrator", "model": self.model,
                  "ms": round((time.perf_counter() - start) * 1000, 1),
                  "turns": getattr(result, "num_turns", None),
                  "cost_usd": getattr(result, "total_cost_usd", None),
                  "rationale": s.get("rationale", "")}]
        trace += [{"agent": c["tool"], "args": c["args"]} for c in tools.calls]
        return TriageResult(
            ticket_id=ticket.id, category=category, category_confidence=confidence,
            priority=priority, escalate=escalate, team=ROUTING[category], sla_minutes=SLA_MINUTES[priority],
            signals=floor.signals + ([] if priority == floor.priority else [f"claude:{priority.value}"]),
            articles=articles[:3], draft_reply=draft, guardrail=report,
            needs_human_review=bool(reasons), review_reasons=reasons, trace=trace,
        )
