from __future__ import annotations

from archeval.agents.router import route
from archeval.agents.system import ARCHITECTURES
from archeval.corpus.contoso import FACTS
from archeval.evaluation.grader import grade
from archeval.evaluation.tasks import TASKS


def task(task_id: str):  # type: ignore[no-untyped-def]
    return next(t for t in TASKS if t.id == task_id)


async def test_single_agent_answers_a_lookup_with_citation(system) -> None:  # type: ignore[no-untyped-def]
    r = await system.run("What is the uptime SLA of Contoso Mail?", ARCHITECTURES["single-large"])
    assert FACTS["Contoso Mail"].sla in r.answer
    assert "[mail-overview]" in r.answer and r.grounded
    assert r.workers == 0 and r.calls >= 2 and r.cost_usd > 0


async def test_multi_agent_delegates_to_parallel_workers(system) -> None:  # type: ignore[no-untyped-def]
    t = task("compare-05")  # 8 products
    r = await system.run(t.question, ARCHITECTURES["multi-mixed"])
    assert r.workers >= 4
    agents = {c["agent"] for c in r.ledger["calls"]}
    assert "lead" in agents and any(a.startswith("worker-") for a in agents)
    tiers = {c["agent"].split("-")[0]: c["tier"] for c in r.ledger["calls"]}
    assert tiers["lead"] == "large" and tiers["worker"] == "small"
    assert grade(t, r).correct


async def test_parallel_workers_shorten_the_critical_path(system) -> None:  # type: ignore[no-untyped-def]
    t = task("compare-05")
    r = await system.run(t.question, ARCHITECTURES["multi-large"])
    serial = sum(c["latency_s"] for c in r.ledger["calls"])
    assert r.latency_s < serial


async def test_router_sends_narrow_questions_to_single_and_wide_to_multi(system) -> None:  # type: ignore[no-untyped-def]
    narrow = await system.run(task("lookup-00").question, ARCHITECTURES["routed"])
    wide = await system.run(task("aggregate-00").question, ARCHITECTURES["routed"])
    assert narrow.ran_as == "single-small" and wide.ran_as == "multi-mixed"
    assert narrow.route_reason and wide.route_reason


def test_router_breadth_counts_products_times_facts() -> None:
    assert route("What is the uptime SLA of Contoso Mail?").architecture == "single"
    assert route("Compare the price and uptime SLA of Contoso Mail and Contoso Drive.").breadth == 4
    assert route("Which Contoso products store data in West Europe?").architecture == "multi"


async def test_failed_model_call_is_a_failed_task_not_a_crash(settings, store) -> None:  # type: ignore[no-untyped-def]
    from archeval.agents.system import AgentSystem
    from archeval.llm.clients import ClientFactory
    from archeval.llm.sim import SimConfig

    tiny = AgentSystem(settings, store, ClientFactory(settings, SimConfig(context_window=200)))
    r = await tiny.run("What is the uptime SLA of Contoso Mail?", ARCHITECTURES["single-large"])
    assert r.error and not r.answer and not r.grounded


async def test_concurrent_runs_keep_separate_ledgers(system) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    a, b = await asyncio.gather(
        system.run(task("lookup-00").question, ARCHITECTURES["single-small"]),
        system.run(task("aggregate-00").question, ARCHITECTURES["multi-large"]))
    assert {c["tier"] for c in a.ledger["calls"]} == {"small"}
    assert a.calls < b.calls
