"""Opinionated Deep Agent graph for deployment."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from deepagents import create_deep_agent
from langchain_core.tools import tool

DEFAULT_MODEL = os.getenv("DEEP_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

SYSTEM_PROMPT = """
You are a rigorous execution-focused deep agent.

Workflow you must follow:
1. Write and maintain a todo list for non-trivial requests.
2. Delegate focused fact-finding to subagents when helpful.
3. Store intermediate drafts in files when the task is long.
4. Before finalizing, run a brief internal critique for risks, gaps, and missing constraints.
5. Return concise, actionable final output.

Quality bar:
- Prefer concrete evidence over assumptions.
- State unresolved uncertainty explicitly.
- Keep output compact unless the user asks for depth.
""".strip()


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


@tool
def confidence_check(claim: str, confidence: float) -> str:
    """Force explicit confidence scoring for key claims.

    Args:
        claim: The claim being assessed.
        confidence: Numeric confidence in [0, 1].
    """
    bounded = min(max(confidence, 0.0), 1.0)
    return f"claim={claim!r}; confidence={bounded:.2f}"


SUBAGENTS = [
    {
        "name": "researcher",
        "description": "Use for evidence collection and source-grounded fact finding.",
        "system_prompt": (
            "You are a focused researcher. Gather evidence, list assumptions, and "
            "report contradictions clearly."
        ),
        "tools": [utc_now, confidence_check],
    },
    {
        "name": "critic",
        "description": "Use for adversarial review of drafts and plans.",
        "system_prompt": (
            "You are a critical reviewer. Find weak logic, untested assumptions, and "
            "missing constraints."
        ),
        "tools": [confidence_check],
    },
]


graph = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, confidence_check],
    system_prompt=SYSTEM_PROMPT,
    subagents=SUBAGENTS,
    interrupt_on={
        "execute": True,
        "write_file": True,
    },
    name="opinionated_deep_agent",
)
