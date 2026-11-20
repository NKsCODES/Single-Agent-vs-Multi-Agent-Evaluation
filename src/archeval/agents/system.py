"""The two architectures under test, built on Microsoft Agent Framework.

Single agent: one agent with the search tool in a loop, context managed by Agent Framework's
``ContextWindowCompactionStrategy`` (older tool results are collapsed, then the oldest turns
are truncated, as the prompt approaches the window).

Orchestrator-worker: a lead agent whose ``research_subtask`` tool starts a fresh worker agent
for each subtask. The lead issues several subtask calls in one turn and Agent Framework runs
them concurrently. Each worker has its own context window and returns compact findings, so the
lead never sees raw search results for the whole task.

Both share the model clients, the search tool, the grounding rules and the iteration budgets.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agent_framework import (
    Agent,
    AgentResponse,
    CharacterEstimatorTokenizer,
    ContextWindowCompactionStrategy,
)
from agent_framework.openai import OpenAIChatCompletionClient

from archeval.agents import prompts
from archeval.agents.router import route
from archeval.agents.tools import make_search_tool
from archeval.config import Settings, Tier
from archeval.llm.clients import ClientFactory
from archeval.llm.ledger import CURRENT_AGENT, CURRENT_LEDGER, CURRENT_WAVE, UsageLedger
from archeval.retrieval.store import HybridStore
from archeval.safety.grounding import check_grounding

Kind = Literal["single", "multi", "routed"]
COMPACTION_PREFIX = "[Tool results: "  # ToolResultCompactionStrategy summary format


def final_text(response: AgentResponse) -> str:
    """The agent's final answer. ``response.text`` would also include the summary messages that
    compaction writes into the history (``[Tool results: ...]``), so take the last assistant
    message that compaction did not create."""
    for msg in reversed(response.messages):
        if msg.role == "assistant" and msg.text and not msg.text.startswith(COMPACTION_PREFIX):
            return msg.text
    return ""


@dataclass(frozen=True)
class Architecture:
    name: str
    kind: Kind
    lead_tier: Tier = "large"
    worker_tier: Tier = "large"


ARCHITECTURES = {
    "single-small": Architecture("single-small", "single", "small", "small"),
    "single-large": Architecture("single-large", "single", "large", "large"),
    "multi-large": Architecture("multi-large", "multi", "large", "large"),
    "multi-mixed": Architecture("multi-mixed", "multi", "large", "small"),
    "routed": Architecture("routed", "routed", "large", "small"),
}
# The router right-sizes both levers: a small single agent for narrow questions, a large lead
# with small workers for breadth.
ROUTED_SINGLE, ROUTED_MULTI = ARCHITECTURES["single-small"], ARCHITECTURES["multi-mixed"]


@dataclass
class AnswerResult:
    question: str
    architecture: str
    ran_as: str
    answer: str
    citations: list[str]
    grounded: bool
    ungrounded: list[str]
    calls: int
    workers: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    wall_s: float
    route_reason: str | None = None
    error: str | None = None
    ledger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentSystem:
    def __init__(self, settings: Settings, store: HybridStore, clients: ClientFactory) -> None:
        self.settings = settings
        self.limits = settings.limits
        self.clients = clients
        self.search = make_search_tool(store, self.limits.search_top_k)

    def _chat(self, tier: Tier, max_iterations: int, concurrent: bool = False
              ) -> OpenAIChatCompletionClient:
        client = self.clients.chat(tier)
        client.function_invocation_configuration.update(  # type: ignore[union-attr]
            {"max_iterations": max_iterations, "allow_concurrent_invocation": concurrent})
        return client

    def _compaction(self) -> ContextWindowCompactionStrategy:
        return ContextWindowCompactionStrategy(
            max_context_window_tokens=self.limits.context_window_tokens,
            max_output_tokens=self.limits.max_output_tokens,
            tokenizer=CharacterEstimatorTokenizer(), keep_last_tool_call_groups=2)

    async def _single(self, question: str, tier: Tier) -> str:
        agent = Agent(client=self._chat(tier, self.limits.single_max_iterations, True),
                      instructions=prompts.SINGLE, tools=[self.search],
                      compaction_strategy=self._compaction())
        CURRENT_AGENT.set("single")
        return final_text(await agent.run(question))

    async def _multi(self, question: str, lead_tier: Tier, worker_tier: Tier) -> str:
        async def research_subtask(instruction: str) -> str:
            """Delegate one focused research subtask to a worker agent with its own context.
            Name the products and the facts to find."""
            ledger = CURRENT_LEDGER.get()
            n, wave = 0, 0
            if ledger is not None:
                ledger.workers_started += 1
                n = ledger.workers_started
                wave = sum(1 for c in ledger.calls if c.agent == "lead")
            a_tok, w_tok = CURRENT_AGENT.set(f"worker-{n}"), CURRENT_WAVE.set(wave)
            try:
                worker = Agent(client=self._chat(worker_tier, self.limits.worker_max_iterations,
                                                 True),
                               instructions=prompts.WORKER, tools=[self.search],
                               compaction_strategy=self._compaction())
                return final_text(await worker.run(instruction[:1000])) or "NOT FOUND | none | none"
            finally:
                CURRENT_AGENT.reset(a_tok)
                CURRENT_WAVE.reset(w_tok)

        lead = Agent(client=self._chat(lead_tier, self.limits.lead_max_iterations, True),
                     instructions=prompts.LEAD.format(max_workers=self.limits.max_workers),
                     tools=[self.search, research_subtask])
        CURRENT_AGENT.set("lead")
        return final_text(await lead.run(question))

    async def run(self, question: str, arch: Architecture) -> AnswerResult:
        ledger = UsageLedger()
        CURRENT_LEDGER.set(ledger)
        ran_as, reason = arch, None
        if arch.kind == "routed":
            r = route(question)
            ran_as = ROUTED_MULTI if r.architecture == "multi" else ROUTED_SINGLE
            reason = r.reason
        started = time.perf_counter()
        error = None
        try:
            if ran_as.kind == "single":
                answer = await self._single(question, ran_as.lead_tier)
            else:
                answer = await self._multi(question, ran_as.lead_tier, ran_as.worker_tier)
        except Exception as exc:  # a failed run is a failed task, not a crashed benchmark
            answer, error = "", f"{type(exc).__name__}: {str(exc)[:200]}"
        grounding = check_grounding(answer, ledger.retrieved_ids)
        return AnswerResult(
            question=question, architecture=arch.name, ran_as=ran_as.name, answer=answer,
            citations=grounding.citations, grounded=grounding.grounded and bool(answer),
            ungrounded=grounding.unknown_citations + grounding.uncited_sentences,
            calls=len(ledger.calls), workers=ledger.workers_started,
            input_tokens=ledger.input_tokens, output_tokens=ledger.output_tokens,
            cost_usd=round(ledger.cost(self.settings.prices), 6),
            latency_s=round(ledger.critical_path_latency(), 3),
            wall_s=round(time.perf_counter() - started, 3), route_reason=reason, error=error,
            ledger=ledger.to_dict(),
        )
