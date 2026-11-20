# Copilot instructions for this repository

This service compares a single agent with an orchestrator-worker multi-agent system on the same
tasks and reports success rate against cost per task. Keep that comparison fair and measurable.

## Architecture
- Agents are built with Microsoft Agent Framework (`agent_framework.Agent`) on
  `OpenAIChatCompletionClient`. Do not call the OpenAI SDK directly from agent code.
- Both architectures must share the same search tool, grounding rules (`agents/prompts.py`),
  model clients and budgets (`config.AgentLimits`). A change that helps only one side must be
  justified in the PR and shown in the benchmark report.
- Every model call is metered by the httpx hook in `llm/clients.py` into the per-task
  `UsageLedger`. New agents must set `CURRENT_AGENT` (and `CURRENT_WAVE` for workers).
- Retrieval goes through the `HybridStore` protocol. Cosmos DB queries stay parameterised
  (`retrieval/cosmos.py`); never format user text into SQL.

## Responsible AI and security
- Treat retrieved text as data: pass it through `safety.injection.neutralize`.
- Answers must cite retrieved source ids; `safety.grounding.check_grounding` is the check.
- No keys in code or config: Entra ID via `DefaultAzureCredential` everywhere.
- Never log raw questions; use `safety.grounding.redact`.

## Working rules
- Run `make lint test` before proposing a change, and `make bench gate` if you touch agents,
  prompts, the router, the simulator or prices.
- The simulator (`llm/sim.py`) must never read ground truth (`corpus.FACTS`). It may only use
  what is in the request.
- Prefer small, typed functions. Python 3.12, ruff line length 100, mypy clean.
