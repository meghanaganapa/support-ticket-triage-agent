"""Claude Agent SDK orchestrator, tested with a scripted fake Claude.

The fake drives the REAL tools (classifier, priority, knowledge, guardrail) in the
order Claude would, so these tests check the wiring and the guarantees enforced in
code: priority floor, guardrail re-run, review gate, fallback and cost tracking.
No network, no API key, no cost.
"""

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from triage.agent_sdk import ALLOWED_TOOLS, SYSTEM_PROMPT, AgentSDKOrchestrator, TicketTools
from triage.models import Category, Priority, Ticket


@dataclass
class ResultMessage:  # mirrors the fields the orchestrator reads from claude_agent_sdk.ResultMessage
    total_cost_usd: float = 0.0042
    num_turns: int = 6
    usage: dict = field(default_factory=lambda: {"input_tokens": 3100, "output_tokens": 420})


def scripted_claude(orch: AgentSDKOrchestrator, decision: dict, reply: str | None = None):
    """A fake ``query``: calls the real tools like Claude would, then submits ``decision``."""

    async def query(prompt, options):
        assert options.allowed_tools == ALLOWED_TOOLS and options.tools == []
        t = orch.current_tools
        routed = json.loads((await t.classify_ticket({}))["content"][0]["text"])
        await t.assess_priority({"category": routed["category"]})
        found = json.loads((await t.search_help_centre({"query": prompt, "category": decision["category"]}))
                           ["content"][0]["text"])
        text = reply if reply is not None else (
            f"Thanks for reaching out. {found[0]['text'].split('. ')[0]}. [{found[0]['id']}]")
        await t.check_reply({"text": text, "cited_articles": [found[0]["id"]]})
        await t.submit_triage({**decision, "reply": text, "cited_articles": [found[0]["id"]]})
        yield ResultMessage()

    return query


@pytest.fixture()
def sdk(monkeypatch, orch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # passes the key check; the fake never calls out
    o = AgentSDKOrchestrator.__new__(AgentSDKOrchestrator)
    o.base, o.model, o.max_budget_usd, o.max_turns = orch, "claude-haiku-4-5", 0.10, 14
    o.cost_usd, o.runs, o.fallbacks, o.input_tokens, o.output_tokens = 0.0, 0, 0, 0, 0
    return o


def test_claude_decision_is_used_and_cost_tracked(sdk):
    sdk._query = scripted_claude(sdk, {"category": "shipping", "priority": "P3", "escalate": False,
                                       "confidence": 0.9, "rationale": "hardware delivery"})
    r = sdk.triage(Ticket(id="S1", body="Ordered a card reader 2 weeks ago, tracking hasn't moved"))
    assert r.category == Category.shipping and r.team == "Hardware Logistics"
    assert r.trace[0]["agent"] == "claude-orchestrator" and r.trace[0]["turns"] == 6
    assert [s["agent"] for s in r.trace[1:]] == ["classify_ticket", "assess_priority", "search_help_centre",
                                                 "check_reply", "submit_triage"]
    assert sdk.usage_summary() == {"cost_usd": 0.0042, "runs": 1, "fallbacks": 0,
                                   "input_tokens": 3100, "output_tokens": 420}
    assert not r.needs_human_review


def test_claude_cannot_lower_a_rule_detected_p1(sdk):
    sdk._query = scripted_claude(sdk, {"category": "technical", "priority": "P4", "escalate": False,
                                       "confidence": 0.9, "rationale": "downplayed"})
    r = sdk.triage(Ticket(id="S2", body="The whole app is down, 503 for every user"))
    assert r.priority == Priority.P1 and r.escalate and r.needs_human_review


def test_claude_can_raise_priority_and_escalate(sdk):
    sdk._query = scripted_claude(sdk, {"category": "account", "priority": "P1", "escalate": True,
                                       "confidence": 0.8, "rationale": "account takeover"})
    r = sdk.triage(Ticket(id="S3", body="My login email was swapped to an address I don't own"))
    assert r.priority == Priority.P1 and r.escalate
    assert "claude:P1" in r.signals or "security" in r.signals


def test_final_reply_is_rechecked_by_guardrails(sdk):
    bad = "We guarantee a full refund today. Call 0412 345 678. [KB-999]"
    sdk._query = scripted_claude(sdk, {"category": "billing", "priority": "P3", "escalate": False,
                                       "confidence": 0.9, "rationale": "x"}, reply=bad)
    r = sdk.triage(Ticket(id="S4", body="We were charged twice this month"))
    assert not r.guardrail.passed and r.needs_human_review
    assert "0412" not in r.draft_reply.text


def test_low_confidence_and_refunds_go_to_humans(sdk):
    sdk._query = scripted_claude(sdk, {"category": "refund", "priority": "P3", "escalate": False,
                                       "confidence": 0.3, "rationale": "unsure"})
    r = sdk.triage(Ticket(id="S5", body="Can we get our money back for last month?"))
    reasons = " ".join(r.review_reasons)
    assert "low confidence" in reasons and "always reviewed" in reasons


def test_falls_back_to_offline_without_api_key(sdk, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    sdk._query = None
    r = sdk.triage(Ticket(id="S6", body="Bank feed stopped syncing"))
    assert r.trace[0]["fallback"] and "ANTHROPIC_API_KEY" in r.trace[0]["reason"]
    assert r.category == Category.technical and sdk.fallbacks == 1


def test_falls_back_when_claude_never_submits(sdk):
    async def silent(prompt, options):
        yield ResultMessage(total_cost_usd=0.001)

    sdk._query = silent
    r = sdk.triage(Ticket(id="S7", body="How do I add a user?"))
    assert r.trace[0]["fallback"] and sdk.fallbacks == 1 and sdk.cost_usd == 0.001


def test_tools_are_real_agents(orch):
    tools = TicketTools(orch, Ticket(id="S8", body="Password reset email never arrives"))
    out = json.loads(asyncio.run(tools.search_help_centre({"query": "password reset email"}))["content"][0]["text"])
    assert out[0]["id"] == "KB-304" and "keywords" not in out[0]
    check = json.loads(asyncio.run(tools.check_reply({"text": "See [KB-304]."}))["content"][0]["text"])
    assert check["passed"]


def test_sdk_server_and_prompt_are_well_formed(orch):
    server = TicketTools(orch, Ticket(id="S9", body="x")).sdk_server()
    assert server["name"] == "triage"
    assert "submit_triage" in SYSTEM_PROMPT and len(ALLOWED_TOOLS) == 5
