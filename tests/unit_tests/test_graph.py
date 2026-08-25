from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from deep_agent import graph

pytestmark = pytest.mark.anyio


async def test_get_agent_returns_read_only_agent_without_execution_runtime() -> None:
    runtime = SimpleNamespace(execution_runtime=None)

    async with graph.get_agent({}, runtime) as agent:
        assert agent is graph.RO_AGENT


async def test_get_agent_builds_sandbox_agent_for_execution_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Mock()
    agent = Mock()
    get_sandbox = AsyncMock(return_value=backend)
    build_agent = Mock(return_value=agent)
    monkeypatch.setattr(graph, "get_or_create_sandbox", get_sandbox)
    monkeypatch.setattr(graph, "_build_agent", build_agent)
    runtime = SimpleNamespace(execution_runtime=Mock())
    config = {"configurable": {"thread_id": "thread-1"}}

    async with graph.get_agent(config, runtime) as result:
        assert result is agent

    get_sandbox.assert_awaited_once_with("thread-1")
    build_agent.assert_called_once_with(backend=backend)


async def test_get_agent_uses_default_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_sandbox = AsyncMock(return_value=Mock())
    monkeypatch.setattr(graph, "get_or_create_sandbox", get_sandbox)
    monkeypatch.setattr(graph, "_build_agent", Mock(return_value=Mock()))
    runtime = SimpleNamespace(execution_runtime=Mock())

    async with graph.get_agent({}, runtime):
        pass

    get_sandbox.assert_awaited_once_with("default")
