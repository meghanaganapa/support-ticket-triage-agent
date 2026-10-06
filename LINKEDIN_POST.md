# LinkedIn post draft

> Edit this into your own voice before posting. Add a screenshot of the CLI output or the architecture diagram; posts with an image get far more reach.

---

My first model scored 99%. It was wrong.

I built a multi-agent system that triages customer support tickets: it routes each ticket to the right team, sets priority and SLA, escalates outages and security incidents, and drafts a reply grounded in the help centre.

The first evaluation looked amazing. Then I noticed that the test tickets used the same phrasings as the training tickets. When I held out whole templates so the model had to handle wording it had never seen, routing accuracy dropped to 58%.

Then I wrote 30 messy tickets by hand ("spinning wheel then a server error", "got an email about a sign in from Brazil"). The classic ML + rules approach caught 0 of 6 urgent incidents.

That's the real lesson: cheap models are brittle on new wording, and that's exactly where LLM agents earn their cost.

How it works:
🔹 5 agents: Classifier → Priority → Knowledge → Responder → Guardrail
🔹 An LLM handles the nuance, cross-checked by an ML model. If they disagree, a human decides.
🔹 The LLM can raise priority but never lower a detected outage
🔹 Deterministic guardrails: PII redaction, no unauthorised refund promises, no hallucinated citations
🔹 Human-in-the-loop: drafts are never auto-sent
🔹 Exposed as an MCP server, so Claude Desktop, Cursor or VS Code can use it as tools
🔹 27 tests, including a CI gate that fails the build if escalation recall drops

Adding a learned escalation model lifted that from 0 to 2 of 6 at zero cost. The LLM path is next to measure.

Code, evaluation and design decisions: https://github.com/meghanaganapa/support-ticket-triage-agent

How do you evaluate your agents? Random splits, or something tougher?

#AIAgents #MultiAgentSystems #MCP #LLM #MachineLearning #Python #GenAI
