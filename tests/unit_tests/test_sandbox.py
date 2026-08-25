from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langsmith.sandbox import ExecutionResult, Snapshot

from deep_agent import sandbox as sandbox_module
from deep_agent.sandbox import (
    DEFAULT_TEMPLATE_FS_CAPACITY,
    LangSmithBackend,
    _ensure_template,
    get_or_create_sandbox,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def remote_sandbox() -> Mock:
    remote = Mock(name="remote-sandbox")
    remote.name = "sandbox-1"
    remote.run = AsyncMock()
    remote.read = AsyncMock()
    remote.write = AsyncMock()
    return remote


@pytest.fixture(autouse=True)
def clear_backend_cache() -> None:
    sandbox_module._backends.clear()


async def test_aexecute_combines_output_and_uses_default_timeout(
    remote_sandbox: Mock,
) -> None:
    remote_sandbox.run.return_value = ExecutionResult(
        stdout="output", stderr="warning", exit_code=3
    )
    backend = LangSmithBackend(remote_sandbox)

    result = await backend.aexecute("command")

    remote_sandbox.run.assert_awaited_once_with("command", timeout=300)
    assert result.output == "output\nwarning"
    assert result.exit_code == 3
    assert result.truncated is False


async def test_aexecute_honors_timeout_and_stderr_only(remote_sandbox: Mock) -> None:
    remote_sandbox.run.return_value = ExecutionResult(
        stdout="", stderr="failed", exit_code=1
    )
    backend = LangSmithBackend(remote_sandbox)

    result = await backend.aexecute("command", timeout=15)

    remote_sandbox.run.assert_awaited_once_with("command", timeout=15)
    assert result.output == "failed"


async def test_awrite_encodes_text(remote_sandbox: Mock) -> None:
    backend = LangSmithBackend(remote_sandbox)

    result = await backend.awrite("/tmp/file.txt", "hello")

    remote_sandbox.write.assert_awaited_once_with("/tmp/file.txt", b"hello")
    assert result.path == "/tmp/file.txt"
    assert result.error is None


async def test_awrite_returns_backend_error(remote_sandbox: Mock) -> None:
    remote_sandbox.write.side_effect = RuntimeError("disk full")
    backend = LangSmithBackend(remote_sandbox)

    result = await backend.awrite("/tmp/file.txt", "hello")

    assert result.path is None
    assert result.error == "Failed to write file '/tmp/file.txt': disk full"


async def test_download_and_upload_files(remote_sandbox: Mock) -> None:
    remote_sandbox.read.side_effect = [b"first", b"second"]
    backend = LangSmithBackend(remote_sandbox)

    downloaded = await backend.adownload_files(["/a", "/b"])
    uploaded = await backend.aupload_files([("/c", b"third"), ("/d", b"fourth")])

    assert [(item.path, item.content, item.error) for item in downloaded] == [
        ("/a", b"first", None),
        ("/b", b"second", None),
    ]
    assert [(item.path, item.error) for item in uploaded] == [
        ("/c", None),
        ("/d", None),
    ]
    assert remote_sandbox.write.await_args_list[0].args == ("/c", b"third")
    assert remote_sandbox.write.await_args_list[1].args == ("/d", b"fourth")


async def test_ensure_template_reuses_exact_snapshot() -> None:
    client = Mock()
    client.list_snapshots = AsyncMock(
        return_value=[
            Snapshot(
                id="snapshot-1",
                name="deep-agent",
                status="ready",
                fs_capacity_bytes=DEFAULT_TEMPLATE_FS_CAPACITY,
            )
        ]
    )
    client.create_snapshot = AsyncMock()

    await _ensure_template(client, "deep-agent", "python:3")

    client.list_snapshots.assert_awaited_once_with(name_contains="deep-agent")
    client.create_snapshot.assert_not_awaited()


async def test_ensure_template_rejects_non_ready_snapshot() -> None:
    client = Mock()
    client.list_snapshots = AsyncMock(
        return_value=[
            Snapshot(
                id="snapshot-1",
                name="deep-agent",
                status="building",
                fs_capacity_bytes=DEFAULT_TEMPLATE_FS_CAPACITY,
            )
        ]
    )
    client.create_snapshot = AsyncMock()

    with pytest.raises(
        RuntimeError, match="Sandbox snapshot 'deep-agent' is not ready: building"
    ):
        await _ensure_template(client, "deep-agent", "python:3")

    client.create_snapshot.assert_not_awaited()


async def test_ensure_template_creates_snapshot_without_exact_match() -> None:
    client = Mock()
    client.list_snapshots = AsyncMock(
        return_value=[
            Snapshot(
                id="snapshot-1",
                name="deep-agent-old",
                status="ready",
                fs_capacity_bytes=DEFAULT_TEMPLATE_FS_CAPACITY,
            )
        ]
    )
    client.create_snapshot = AsyncMock()

    await _ensure_template(client, "deep-agent", "python:3")

    client.create_snapshot.assert_awaited_once_with(
        name="deep-agent",
        docker_image="python:3",
        fs_capacity_bytes=DEFAULT_TEMPLATE_FS_CAPACITY,
    )


async def test_get_or_create_sandbox_uses_snapshot_and_caches_backend(
    monkeypatch: pytest.MonkeyPatch,
    remote_sandbox: Mock,
) -> None:
    client = Mock()
    client.create_sandbox = AsyncMock(return_value=remote_sandbox)
    client_factory = Mock(return_value=client)
    ensure_template = AsyncMock()
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("SANDBOX_TEMPLATE_NAME", "custom-template")
    monkeypatch.setenv("SANDBOX_TEMPLATE_IMAGE", "python:3.14")
    monkeypatch.setattr(sandbox_module, "AsyncSandboxClient", client_factory)
    monkeypatch.setattr(sandbox_module, "_ensure_template", ensure_template)

    backend = await get_or_create_sandbox("thread-1")
    cached = await get_or_create_sandbox("thread-1")

    assert backend is cached
    assert backend.id == "sandbox-1"
    client_factory.assert_called_once_with(api_key="test-key")
    ensure_template.assert_awaited_once_with(client, "custom-template", "python:3.14")
    client.create_sandbox.assert_awaited_once_with(
        snapshot_name="custom-template", timeout=180
    )


async def test_get_or_create_sandbox_falls_back_to_prod_api_key(
    monkeypatch: pytest.MonkeyPatch,
    remote_sandbox: Mock,
) -> None:
    client = SimpleNamespace(create_sandbox=AsyncMock(return_value=remote_sandbox))
    client_factory = Mock(return_value=client)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY_PROD", "prod-key")
    monkeypatch.setattr(sandbox_module, "AsyncSandboxClient", client_factory)
    monkeypatch.setattr(sandbox_module, "_ensure_template", AsyncMock())

    await get_or_create_sandbox("thread-1")

    client_factory.assert_called_once_with(api_key="prod-key")
