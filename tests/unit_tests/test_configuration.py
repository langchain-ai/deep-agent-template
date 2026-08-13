from langgraph.pregel import Pregel

from deep_agent.graph import RO_AGENT, SUBAGENTS, SYSTEM_PROMPT
from deep_agent.sandbox import sandbox_name


def test_graph_compiles() -> None:
    assert isinstance(RO_AGENT, Pregel)


def test_subagents_configured() -> None:
    names = {item["name"] for item in SUBAGENTS}
    assert names == {"researcher", "critic"}


def test_system_prompt_is_nonempty() -> None:
    assert len(SYSTEM_PROMPT.strip()) > 0


def test_sandbox_name_is_thread_scoped() -> None:
    assert sandbox_name("abc") == "thread-abc"
