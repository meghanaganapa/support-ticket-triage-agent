from conftest import FakeLLM

from triage.agents import ClassifierAgent, KnowledgeAgent, PriorityAgent
from triage.data import load_rows, split, to_ticket
from triage.llm import OfflineLLM
from triage.models import Category, Priority, Ticket

TRAIN, _ = split(load_rows())


def trained_classifier(llm):
    return ClassifierAgent(llm).fit([to_ticket(r) for r in TRAIN], [r["category"] for r in TRAIN])


# ---------- classifier ----------

def test_classifier_offline_routes_obvious_ticket():
    result = trained_classifier(OfflineLLM()).run(Ticket(id="1", body="The card reader we ordered never arrived"))
    assert result.category == Category.shipping


def test_classifier_uses_llm_when_it_agrees_with_baseline():
    llm = FakeLLM(replies=[{"category": "shipping", "confidence": 0.95, "rationale": "hardware delivery"}])
    result = trained_classifier(llm).run(Ticket(id="1", body="The card reader we ordered never arrived"))
    assert result.category == Category.shipping and result.confidence == 0.95


def test_classifier_disagreement_lowers_confidence():
    llm = FakeLLM(replies=[{"category": "feature_request", "confidence": 0.9}])
    result = trained_classifier(llm).run(Ticket(id="1", body="The card reader we ordered never arrived"))
    assert result.confidence <= 0.5
    assert "ML baseline suggested" in result.rationale


def test_classifier_falls_back_on_malformed_llm_output():
    llm = FakeLLM(replies=[{"category": "not-a-category"}])
    result = trained_classifier(llm).run(Ticket(id="1", body="The card reader we ordered never arrived"))
    assert result.category == Category.shipping


# ---------- priority ----------

def test_rules_escalate_outage():
    pa = PriorityAgent(OfflineLLM()).rules(Ticket(id="1", body="The app is down, 503 for everyone"), Category.technical)
    assert pa.priority == Priority.P1 and pa.escalate and "outage" in pa.signals


def test_rules_escalate_security():
    pa = PriorityAgent(OfflineLLM()).rules(Ticket(id="1", body="I think we've been hacked"), Category.account)
    assert pa.priority == Priority.P1 and pa.escalate


def test_rules_escalate_legal_threat_as_p2():
    pa = PriorityAgent(OfflineLLM()).rules(Ticket(id="1", body="Refund me or I'll call my lawyer"), Category.refund)
    assert pa.priority == Priority.P2 and pa.escalate


def test_feature_request_defaults_to_p4_even_with_deadline_word():
    pa = PriorityAgent(OfflineLLM()).rules(Ticket(id="1", body="Add dark mode today please"), Category.feature_request)
    assert pa.priority == Priority.P4 and not pa.escalate


def test_llm_can_raise_but_never_lower_priority():
    ticket = Ticket(id="1", body="Everything is down, 503 errors")
    lower = PriorityAgent(FakeLLM(replies=[{"priority": "P4", "escalate": False}])).run(ticket, Category.technical)
    assert lower.priority == Priority.P1 and lower.escalate

    calm = Ticket(id="2", body="Small question about my invoice layout")
    raised = PriorityAgent(FakeLLM(replies=[{"priority": "P2", "escalate": True, "signals": ["x"]}])).run(
        calm, Category.billing)
    assert raised.priority == Priority.P2 and raised.escalate and "llm:x" in raised.signals


# ---------- knowledge ----------

def test_knowledge_finds_bank_feed_article():
    arts = KnowledgeAgent().run(Ticket(id="1", body="Bank feed stopped syncing transactions"), Category.technical)
    assert arts[0].id == "KB-201"


def test_knowledge_category_boost():
    arts = KnowledgeAgent().search("refund", category=Category.refund, k=1)
    assert arts[0].id.startswith("KB-1")
