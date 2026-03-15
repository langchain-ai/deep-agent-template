from langgraph.pregel import Pregel

from deep_agent.graph import SUBAGENTS, SYSTEM_PROMPT, graph


def test_graph_compiles() -> None:
    assert isinstance(graph, Pregel)


def test_subagents_configured() -> None:
    names = {item["name"] for item in SUBAGENTS}
    assert names == {"researcher", "critic"}


def test_system_prompt_is_nonempty() -> None:
    assert len(SYSTEM_PROMPT.strip()) > 0
