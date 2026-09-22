"""Thread-scoped LangSmith sandbox resolution.

The sandbox for a thread is identified by its name (``thread-<thread_id>``) so
that any instance of a scaled deployment resolves the same sandbox. Nothing is
cached in process memory.

See https://docs.langchain.com/oss/python/deepagents/sandboxes for details.
"""

from __future__ import annotations

import os

from deepagents.backends.langsmith import LangSmithSandbox
from langsmith.sandbox import (
    AsyncSandbox,
    AsyncSandboxClient,
    ResourceAlreadyExistsError,
    ResourceNameConflictError,
    ResourceNotFoundError,
)

DEFAULT_IDLE_TTL_SECONDS = 3600


def _idle_ttl_seconds() -> int:
    return int(os.environ.get("SANDBOX_IDLE_TTL_SECONDS", DEFAULT_IDLE_TTL_SECONDS))


def sandbox_name(thread_id: str) -> str:
    """Return the deterministic sandbox name for a thread."""
    return f"thread-{thread_id}"


async def get_or_create_sandbox(thread_id: str) -> LangSmithSandbox:
    """Resolve the sandbox backend for a thread, creating the sandbox if needed."""
    client = AsyncSandboxClient()
    name = sandbox_name(thread_id)

    try:
        sandbox = await client.get_sandbox(name)
    except ResourceNotFoundError:
        sandbox = await _create_sandbox(client, name)

    if sandbox.status == "stopped":
        await sandbox.start()

    return LangSmithSandbox(sandbox=sandbox.to_sync())


async def _create_sandbox(client: AsyncSandboxClient, name: str) -> AsyncSandbox:
    """Create the named sandbox, tolerating a concurrent creation of the same name."""
    try:
        return await client.create_sandbox(
            name=name,
            idle_ttl_seconds=_idle_ttl_seconds(),
        )
    except (ResourceAlreadyExistsError, ResourceNameConflictError):
        return await client.get_sandbox(name)
