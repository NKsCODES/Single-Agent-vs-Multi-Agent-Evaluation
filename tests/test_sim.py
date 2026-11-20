from __future__ import annotations

import json

import httpx

from archeval.llm.sim import SimConfig, SimLLM, parse_task


def call(sim: SimLLM, messages: list[dict], tools: bool = True,  # type: ignore[type-arg]
         model: str = "large") -> httpx.Response:
    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "search_knowledge_base",
                                                           "parameters": {}}}]
    req = httpx.Request("POST", "http://sim.local/v1/chat/completions", content=json.dumps(body))
    return sim.handle(req)


def test_parse_task_kinds() -> None:
    assert parse_task("What is the uptime SLA of Contoso Mail?").kind == "lookup"
    assert parse_task("Compare the price of Contoso Mail and Contoso Chat.").kind == "compare"
    assert parse_task("What is the price of the product that introduced Smart Sync?").kind == \
        "multihop"
    t = parse_task("Which Contoso products have an uptime SLA of 99.9% and store data in "
                   "West Europe?")
    assert t.kind == "aggregate" and t.region == "West Europe"


def test_sim_searches_then_answers_from_context_only() -> None:
    sim = SimLLM()
    q = [{"role": "system", "content": "[agent: single]"},
         {"role": "user", "content": "What is the uptime SLA of Contoso Mail?"}]
    first = call(sim, q).json()["choices"][0]["message"]
    assert first["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
    no_tools = call(sim, q, tools=False).json()["choices"][0]["message"]["content"]
    assert "not found" in no_tools.lower() or "99" in no_tools


def test_sim_recall_degrades_with_context_and_floors() -> None:
    sim = SimLLM(SimConfig())
    assert sim.recall("large", 1000) == 0.97
    assert sim.recall("large", 6000) < sim.recall("large", 3000)
    assert sim.recall("small", 6000) < sim.recall("large", 6000)
    assert sim.recall("small", 100_000) == 0.35


def test_sim_rejects_prompts_over_the_context_window() -> None:
    sim = SimLLM(SimConfig(context_window=50))
    r = call(sim, [{"role": "user", "content": "x" * 1000}])
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"


def test_sim_is_deterministic_and_reports_usage() -> None:
    q = [{"role": "user", "content": "What is the uptime SLA of Contoso Mail?"}]
    a, b = call(SimLLM(), q).json(), call(SimLLM(), q).json()
    assert a["choices"] == b["choices"] and a["usage"]["prompt_tokens"] > 0
