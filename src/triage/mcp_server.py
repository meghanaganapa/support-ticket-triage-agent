"""MCP server: lets any MCP client (Claude Desktop, Claude Code, Cursor, VS Code)
use the triage system as tools.

A support lead can then ask their assistant things like:
  "Triage this email and tell me who should handle it"
  "Find the help article about bank feeds"
  "What's the SLA for a P2?"

Run (stdio transport, which desktop clients use):
    python -m triage.mcp_server
"""

from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from .factory import build_orchestrator
from .models import ROUTING, SLA_MINUTES, Category, Priority, Ticket

server = MCPServer(
    name="support-triage",
    instructions=("Tools for triaging CloudLedger customer support tickets: route to a team, "
                  "set priority and SLA, find help-centre articles, and draft a grounded reply."),
)


@lru_cache(maxsize=1)
def _orch():
    return build_orchestrator()


@server.tool()
def triage_ticket(body: str, subject: str = "", customer: str = "Unknown customer") -> dict:
    """Run the full multi-agent pipeline on one ticket. Returns team, priority, SLA,
    escalation flag, help articles, a guardrail-checked draft reply and review reasons."""
    result = _orch().triage(Ticket(id="MCP-1", subject=subject, body=body, customer=customer))
    return result.model_dump(mode="json", exclude={"trace"})


@server.tool()
def search_help_centre(query: str, k: int = 3) -> list[dict]:
    """Search the help centre and return the most relevant articles with scores."""
    orch = _orch()
    knowledge = orch.base.knowledge if hasattr(orch, "base") else orch.knowledge
    return [a.model_dump(exclude={"keywords"}) for a in knowledge.search(query, k=k)]


@server.tool()
def routing_policy() -> dict:
    """Return which team owns each category and the first-response SLA for each priority."""
    return {
        "teams": {c.value: ROUTING[c] for c in Category},
        "sla_minutes": {p.value: SLA_MINUTES[p] for p in Priority},
    }


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
