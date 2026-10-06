from triage.agents.guardrail import GuardrailAgent
from triage.models import DraftReply, KBArticle

ARTICLES = [KBArticle(id="KB-102", title="Duplicate charges", text="...")]


def run(text: str, cited=None, customers=None, customer="Acme Pty Ltd"):
    agent = GuardrailAgent(known_customers=customers or ["Acme Pty Ltd", "Bluegum Cafe"])
    return agent.run(DraftReply(text=text, cited_articles=cited or []), ARTICLES, customer)


def test_clean_grounded_reply_passes():
    draft, report = run("Duplicate charges usually drop off in 3-5 days [KB-102].")
    assert report.passed
    assert draft.cited_articles == ["KB-102"]


def test_redacts_card_number_and_phone():
    draft, report = run("Your card 4111 1111 1111 1111 and 0412 345 678 are on file [KB-102].")
    assert "4111" not in draft.text and "0412" not in draft.text
    assert report.redactions == 2


def test_keeps_allow_listed_support_email():
    draft, report = run("Emails come from no-reply@cloudledger.example [KB-102].")
    assert "no-reply@cloudledger.example" in draft.text
    assert report.redactions == 0


def test_blocks_unauthorised_promises():
    _, report = run("We guarantee a fix and we will refund you today [KB-102].")
    assert not report.passed
    assert sum("unauthorised promise" in i for i in report.issues) == 2


def test_blocks_hallucinated_citation():
    _, report = run("See [KB-999] for details.")
    assert not report.passed
    assert any("not retrieved" in i for i in report.issues)


def test_requires_grounding_when_articles_exist():
    _, report = run("Thanks, we're looking into it.")
    assert any("not grounded" in i for i in report.issues)


def test_blocks_other_customer_names():
    _, report = run("Like Bluegum Cafe, you were charged twice [KB-102].")
    assert any("another customer" in i for i in report.issues)
