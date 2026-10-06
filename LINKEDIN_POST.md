# LinkedIn post drafts

> Edit these into your own voice before posting. Add an image (CLI output, the architecture diagram, or the v0.1 → v0.2 table); posts with an image get far more reach. Post 1 first; post 2 a week or so later as the follow-up. Every number below matches `reports/`.

---

## Post 1: the v0.1 story

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
🔹 A CI gate that fails the build if escalation recall drops

Adding a learned escalation model lifted that from 0 to 2 of 6 at zero cost.

Code, evaluation and design decisions: https://github.com/meghanaganapa/support-ticket-triage-agent

How do you evaluate your agents? Random splits, or something tougher?

#AIAgents #MultiAgentSystems #MCP #LLM #MachineLearning #Python #GenAI

---

## Post 2: the v0.2 follow-up

"+17.8 points" means nothing unless you can show it isn't luck.

v0.2 of my support-ticket triage agents is out. Before changing a single model, I wrote a 90-ticket test set and committed it to git. It's checksum-locked, CI fails if it changes, and I never tuned on it.

On that locked set, v0.1 → v0.2:
✅ Routing accuracy: 80.0% → 86.7%
✅ Tickets handled without a human: 44.4% → 62.2%
✅ Right help article in the top 2: 78.3% → 88.4%

All three held up under a paired bootstrap: the 95% interval of the difference excludes zero.

And the honest part:
⚠️ Priority accuracy and escalation recall improved, but within noise. With 14 urgent tickets, one ticket is 7 points.
⚠️ Escalation precision went down (78.6% → 70.6%), a trade-off I chose on purpose: a missed outage costs more than a false alarm.

What actually moved the numbers:
🔹 Calibrated confidence, so the "send to a human" threshold is a business rule, not a magic number
🔹 102 varied training tickets, with a leakage test so none copy the test set
🔹 Keywords on help articles for better retrieval

The architecture also changed: Claude now orchestrates the agents as tools through the Claude Agent SDK. Prompts are requests, not controls, so the guarantees live in code. Claude can raise priority but never lower an outage, every reply is re-checked, and refunds always reach a person. Each ticket has a budget cap, and if Claude fails, the free offline system takes over.

Next: measuring the Claude-orchestrated version on the same locked set, accuracy and cost per ticket.

https://github.com/meghanaganapa/support-ticket-triage-agent

What's the most convincing evidence you've seen that an AI system actually improved?

#AIAgents #ClaudeAgentSDK #MCP #LLMOps #MachineLearning #Evaluation #Python
