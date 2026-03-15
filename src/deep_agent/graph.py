"""Deep Agent graph for deployment."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
from deepagents import create_deep_agent
from langchain_core.tools import tool

DEFAULT_MODEL = os.getenv("DEEP_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

_http_client = httpx.AsyncClient(
    headers={"User-Agent": "deep-agent/0.1"},
    timeout=10,
    follow_redirects=True,
)

SYSTEM_PROMPT = """
You are a deep agent.

Workflow:
1. Write and maintain a todo list for non-trivial requests.
2. Delegate focused fact-finding to subagents when helpful.
3. Store intermediate drafts in files when the task is long.
4. Before finalizing, critique your work for risks, gaps, and missing constraints.
5. Return concise, actionable output.

- Prefer concrete evidence over assumptions.
- State unresolved uncertainty explicitly.
- Keep output compact unless the user asks for depth.
""".strip()


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


@tool
async def web_fetch(url: str) -> str:
    """Fetch a URL and return its body as text.

    Args:
        url: The URL to fetch (must be http or https).
    """
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        resp = await _http_client.get(url)
        resp.raise_for_status()
        return resp.text[:50_000]
    except httpx.HTTPError as exc:
        return f"Error fetching {url}: {exc}"


SUBAGENTS = [
    {
        "name": "researcher",
        "description": "Use for evidence collection and source-grounded fact finding.",
        "system_prompt": (
            "You are a focused researcher. Gather evidence, list assumptions, and "
            "report contradictions clearly."
        ),
        "tools": [utc_now, web_fetch],
    },
    {
        "name": "critic",
        "description": "Use for adversarial review of drafts and plans.",
        "system_prompt": (
            "You are a critical reviewer. Find weak logic, untested assumptions, and "
            "missing constraints."
        ),
        "tools": [utc_now],
    },
]


graph = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, web_fetch],
    system_prompt=SYSTEM_PROMPT,
    subagents=SUBAGENTS,
    interrupt_on={
        "execute": True,
        "write_file": True,
    },
    name="deep_agent",
)
