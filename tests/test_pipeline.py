import asyncio
import difflib

from fastapi.testclient import TestClient

from triage import api, mcp_server
from triage.data import DEV_DATA, EXTRA_TRAIN, TEST_DATA, load_rows, split, verify_test_set
from triage.evaluate import evaluate
from triage.models import Priority, Ticket


def test_end_to_end_outage_is_escalated_and_reviewed(orch):
    r = orch.triage(Ticket(id="X1", body="Nobody can log in, the whole app is down with a 503 error"))
    assert r.priority == Priority.P1 and r.escalate
    assert r.needs_human_review
    assert r.sla_minutes == 15
    assert [s["agent"] for s in r.trace] == ["classifier", "priority", "knowledge", "responder", "guardrail"]


def test_refunds_always_get_human_review(orch):
    r = orch.triage(Ticket(id="X2", body="We cancelled last week but were still charged $49. Please refund it."))
    assert r.category.value == "refund"
    assert any("always reviewed" in reason for reason in r.review_reasons)


def test_offline_drafts_are_grounded_and_pass_guardrails(orch):
    r = orch.triage(Ticket(id="X3", body="How do I add a new user to our account?"))
    assert r.guardrail.passed
    assert r.draft_reply.cited_articles and r.draft_reply.cited_articles[0] in r.draft_reply.text


def test_split_holds_out_whole_templates():
    train, test = split(load_rows())
    assert not {r["template_id"] for r in train} & {r["template_id"] for r in test}
    assert {r["category"] for r in train} == {r["category"] for r in test}


def test_dev_and_locked_test_sets_are_labelled():
    for path, n in ((DEV_DATA, 30), (TEST_DATA, 90)):
        rows = load_rows(path)
        assert len(rows) == n
        assert all({"category", "priority", "escalate", "kb"} <= r.keys() for r in rows)


def test_locked_test_set_is_unchanged():
    verify_test_set()  # raises if data/test_locked.jsonl no longer matches its committed checksum


def test_no_training_ticket_near_duplicates_an_evaluation_ticket():
    train = [r["body"].lower() for r in load_rows(EXTRA_TRAIN)]
    held = [r["body"].lower() for r in load_rows(DEV_DATA) + load_rows(TEST_DATA)]
    worst = max(difflib.SequenceMatcher(None, a, b).ratio() for a in train for b in held)
    assert worst < 0.8, f"possible leakage: similarity {worst:.2f}"


def test_evaluation_regression_gate():
    """CI fails if a change makes the offline system worse on the sets it was tuned on."""
    held = evaluate("offline", dataset="heldout")
    assert held["escalation_recall"]["value"] >= 0.95
    assert held["routing_accuracy"]["value"] >= 0.65
    assert held["priority_accuracy"]["value"] >= 0.85
    assert held["auto_queued_routing_accuracy"]["value"] >= 0.95
    assert held["guardrail_pass_rate"]["value"] == 1.0
    dev = evaluate("offline", dataset="dev")
    assert dev["routing_accuracy"]["value"] >= 0.85
    assert dev["retrieval_hit_at_2"]["value"] >= 0.9


def test_api_health_and_triage():
    client = TestClient(api.app)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["backend"] == "TriageOrchestrator"
    resp = client.post("/triage", json={"id": "A1", "body": "Our card was billed twice this month"})
    assert resp.status_code == 200
    assert resp.json()["team"] in {"Billing Ops", "Tier-2 Engineering", "Account Support"}


def test_mcp_server_exposes_tools():
    tools = asyncio.run(mcp_server.server.list_tools())
    assert {t.name for t in tools} == {"triage_ticket", "search_help_centre", "routing_policy"}


def test_mcp_tools_return_data():
    out = mcp_server.triage_ticket(body="Bank feed for ANZ stopped syncing since Monday")
    assert out["category"] == "technical" and "trace" not in out
    assert mcp_server.search_help_centre("password reset email")[0]["id"] == "KB-304"
    assert mcp_server.routing_policy()["sla_minutes"]["P1"] == 15
