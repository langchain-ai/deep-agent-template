# Deep Agent Template (Opinionated)

Opinionated deployment template for a Deep Agent built with `create_deep_agent(...)`.

## What this template gives you

- A deployable Deep Agent graph at `src/deep_agent/graph.py`.
- Explicit workflow prompt (plan, delegate, critique, finalize).
- Two predefined subagents (`researcher`, `critic`).
- Human-in-the-loop interrupts on `execute` and `write_file`.
- A `uv`-managed local workflow with a small `Makefile` wrapper and starter tests.

## Quickstart

1. Sync the project with `uv`:

```bash
uv sync
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Run locally:

```bash
uv run langgraph dev
```

Optional `make` wrappers:

```bash
make dev
make run
```

## Tests and lint

```bash
make test
make integration-tests
make lint
make format
```

Integration tests are skipped unless `ANTHROPIC_API_KEY` is set.

## Deploy to LangSmith

1. Push this template to a Git repository.
2. In LangSmith, create a new Deployment from that repo.
3. Set environment variables for your selected model provider and optional tracing key.
4. Deploy using the provided `langgraph.json`.

## Reference docs

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents quickstart: https://docs.langchain.com/oss/python/deepagents/quickstart
- LangGraph deployment in LangSmith: https://docs.langchain.com/oss/python/langchain/deploy
