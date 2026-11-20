"""An OpenAI-compatible model simulator for offline, reproducible experiments.

Why it exists: the comparison has to run in CI, at zero cost, with the same result every time.
The simulator answers ``POST /v1/chat/completions`` exactly like the API, so the production
code path (Microsoft Agent Framework -> OpenAI SDK -> HTTP) runs unchanged; only the transport
is swapped. Point the same code at Azure OpenAI to measure a real model.

What it models, explicitly and with tunable parameters:

* **Reading.** The model only knows facts that are in its context window: retrieved chunks or
  findings passed to it. It extracts them with the same patterns a reader would use.
* **Context degradation.** The chance of noticing a fact falls as the prompt grows past a knee
  (the "lost in the middle" or context-rot effect), more steeply for the small tier. A fact
  handed over as a one-line structured finding is easier to read than the same fact inside a
  retrieved passage, so its miss chance is scaled by ``finding_miss_factor``.
* **Hallucination.** When a fact is missing at answer time, the model sometimes guesses a
  plausible value instead of saying it does not know.
* **Context limit.** Prompts over the window get the same 400 error the real API returns.
* **Budgets** are not in the simulator: the agent framework enforces iteration limits, so the
  simulator just answers with what it knows when tools are withdrawn.

It has no access to the ground truth. Everything it "knows" comes from the request.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx2 as httpx  # the HTTP client the OpenAI SDK v3 is built on

from archeval.corpus.contoso import ATTRIBUTES

PRODUCT_RE = re.compile(r"\bContoso (?:[A-Z][A-Za-z]+)\b")
CHUNK_RE = re.compile(r"\[(?P<id>[a-z0-9][a-z0-9-]*)\]\s+(?P<text>.*?)"
                      r"(?=\s*\[[a-z0-9][a-z0-9-]*\]\s|\Z)", re.S)
FINDING_RE = re.compile(r"FINDING \| (?P<e>Contoso [A-Za-z]+) \| (?P<a>[a-z0-9_]+) \| "
                        r"(?P<v>[^|\n]+?) \| (?P<c>[a-z0-9-]+|none)")
EXTRACT = {
    "sla": re.compile(r"backed by a (\d{2}\.\d+%) uptime SLA"),
    "regions": re.compile(r"stored in these regions: ([^.]+)\."),
    "price": re.compile(r"Pricing starts at (\$\d+) per user per month"),
    "max_users": re.compile(r"supports up to ([\d,]+) users"),
    "encryption": re.compile(r"encrypted at rest with ([^.]+)\."),
    "p1_response": re.compile(r"P1 incidents receive a first response within ([^.]+)\."),
}
RELEASE_RE = re.compile(r"Version (\d+\.\d+) \((\d{4}-\d{2}-\d{2})\) introduced ([^.]+)\.")
CATALOG_RE = re.compile(r"offers these cloud products: ([^.]+)\.")
ATTR_KEYWORDS = [
    ("sla", ("uptime sla", "sla", "uptime")),
    ("regions", ("data residency", "regions", "region", "store data")),
    ("price", ("price per user", "pricing", "price")),
    ("max_users", ("maximum users", "users per tenant", "max users")),
    ("encryption", ("encryption",)),
    ("p1_response", ("p1", "response time")),
]
DOC_HINT = {"sla": "overview", "regions": "overview", "price": "overview",
            "max_users": "overview", "encryption": "security", "p1_response": "support policy"}
GUESS = {"sla": "99.9%", "regions": "East US", "price": "$10", "max_users": "10,000",
         "encryption": "AES-256", "p1_response": "1 hour"}
AGG_RE = re.compile(r"Which Contoso products have an uptime SLA of ([\d.]+%) and store data in "
                    r"([A-Z][a-z]+ [A-Z][a-z]+)\?")
MULTIHOP_RE = re.compile(r"the product that introduced (.+?)\?")


@dataclass
class SimConfig:
    context_window: int = 16_000
    knee_tokens: int = 2_000
    base_recall: dict[str, float] = field(default_factory=lambda: {"large": 0.97, "small": 0.92})
    recall_slope_per_1k: dict[str, float] = field(
        default_factory=lambda: {"large": 0.025, "small": 0.045})
    recall_floor: float = 0.35
    hallucination_rate: dict[str, float] = field(
        default_factory=lambda: {"large": 0.05, "small": 0.15})
    finding_miss_factor: float = 0.2
    parallel_searches: int = 2
    max_query_attempts: int = 3
    lead_followup_waves: int = 1
    seed: str = "archeval-v1"
    # modeled latency in seconds: base + per input token + per output token
    latency: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        "large": (0.50, 0.00004, 0.020), "small": (0.30, 0.00002, 0.008)})


@dataclass
class Task:
    kind: str  # lookup | compare | multihop | aggregate
    entities: list[str]
    attrs: list[str]
    feature: str | None = None
    sla: str | None = None
    region: str | None = None


def parse_task(text: str) -> Task:
    m = AGG_RE.search(text)
    if m:
        return Task("aggregate", [], ["sla", "regions"], sla=m.group(1), region=m.group(2))
    attrs = parse_attrs(text)
    m = MULTIHOP_RE.search(text)
    if m:
        return Task("multihop", [], attrs, feature=m.group(1).strip())
    ents = list(dict.fromkeys(PRODUCT_RE.findall(text)))
    return Task("lookup" if len(ents) <= 1 else "compare", ents, attrs)


def parse_attrs(text: str) -> list[str]:
    low = text.lower()
    found = []
    for key, words in ATTR_KEYWORDS:
        if any(w in low for w in words):
            found.append(key)
    return found


def _text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def _tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))


@dataclass
class Known:
    facts: dict[str, dict[str, tuple[str, str]]] = field(default_factory=lambda: defaultdict(dict))
    features: dict[str, tuple[str, str]] = field(default_factory=dict)  # feature -> (product, id)
    catalog: list[str] | None = None
    catalog_id: str | None = None
    not_found: set[tuple[str, str]] = field(default_factory=set)


class SimLLM:
    def __init__(self, config: SimConfig | None = None,
                 tiers: dict[str, str] | None = None) -> None:
        self.cfg = config or SimConfig()
        self.tiers = tiers or {"large": "large", "small": "small"}  # deployment -> tier

    # ------------------------------------------------------------------ http entry point

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        body = json.loads(request.content)
        tier = self.tiers.get(body.get("model", ""), "large")
        messages = body.get("messages", [])
        tools = body.get("tools") or []
        prompt_tokens = _tokens(json.dumps(messages)) + _tokens(json.dumps(tools))
        if prompt_tokens > self.cfg.context_window:
            return httpx.Response(400, json={"error": {
                "message": f"This model's maximum context length is {self.cfg.context_window} "
                           f"tokens. Your messages resulted in {prompt_tokens} tokens.",
                "type": "invalid_request_error", "code": "context_length_exceeded"}})
        tool_names = {t["function"]["name"] for t in tools if t.get("type") == "function"}
        if body.get("tool_choice") == "none":
            tool_names = set()
        content, calls = self._decide(tier, messages, tool_names, prompt_tokens)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            turn = sum(1 for m in messages if m.get("role") == "assistant") + 1
            message["tool_calls"] = [
                {"id": f"call_{turn}_{i}", "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(calls)]
        completion_tokens = _tokens(json.dumps(message))
        base, per_in, per_out = self.cfg.latency[tier]
        latency = base + per_in * prompt_tokens + per_out * completion_tokens
        return httpx.Response(200, headers={"x-sim-latency": f"{latency:.4f}"}, json={
            "id": "sim-" + hashlib.sha1(request.content).hexdigest()[:12],
            "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        })

    # ------------------------------------------------------------------ reading

    def recall(self, tier: str, prompt_tokens: int) -> float:
        over = max(0, prompt_tokens - self.cfg.knee_tokens) / 1000
        p = self.cfg.base_recall[tier] - self.cfg.recall_slope_per_1k[tier] * over
        return max(self.cfg.recall_floor, p)

    def _roll(self, *parts: object) -> float:
        h = hashlib.sha256("|".join(map(str, (self.cfg.seed, *parts))).encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def _read(self, messages: list[dict[str, Any]], p: float, who: str) -> Known:
        k = Known()
        p_finding = 1 - (1 - p) * self.cfg.finding_miss_factor
        for mi, msg in enumerate(messages):
            if msg.get("role") == "system":
                continue
            text = _text(msg)
            for fm in FINDING_RE.finditer(text):
                e, a, v, c = fm.group("e"), fm.group("a"), fm.group("v").strip(), fm.group("c")
                if self._roll(who, "finding", e, a, mi) < p_finding:
                    if a == "introduced":
                        k.features[v.lower()] = (e, c)
                    else:
                        k.facts[e][a] = (v, c)
            for nf in re.finditer(r"NOT FOUND \| (Contoso [A-Za-z]+) \| ([a-z0-9_]+)", text):
                k.not_found.add((nf.group(1), nf.group(2)))
            for cm in CHUNK_RE.finditer(text):
                cid, chunk = cm.group("id"), cm.group("text")
                cat = CATALOG_RE.search(chunk)
                if cat and self._roll(who, "catalog", mi) < p:
                    k.catalog = [x.strip() for x in cat.group(1).split(",")]
                    k.catalog_id = cid
                lead = PRODUCT_RE.match(chunk)
                if not lead:
                    continue
                product = lead.group(0)
                for attr, rx in EXTRACT.items():
                    m = rx.search(chunk)
                    if m and self._roll(who, product, attr, mi) < p:
                        k.facts[product][attr] = (m.group(1).strip(), cid)
                for rm in RELEASE_RE.finditer(chunk):
                    if self._roll(who, "release", rm.group(3), mi) < p:
                        k.features[rm.group(3).lower()] = (product, cid)
        return k

    # ------------------------------------------------------------------ deciding

    def _decide(self, tier: str, messages: list[dict[str, Any]], tools: set[str],
                prompt_tokens: int) -> tuple[str | None, list[tuple[str, dict[str, Any]]]]:
        system = next((_text(m) for m in messages if m.get("role") == "system"), "")
        task_text = next((_text(m) for m in messages if m.get("role") == "user"), "")
        role = "lead" if "[agent: lead]" in system else (
            "worker" if "[agent: worker]" in system else "single")
        who = (role, hashlib.sha1(task_text.encode()).hexdigest()[:10],
               sum(1 for m in messages if m.get("role") == "assistant"))
        known = self._read(messages, self.recall(tier, prompt_tokens), "|".join(map(str, who)))
        task = parse_task(task_text)
        if role == "lead":
            return self._lead(tier, task, task_text, system, messages, known, tools, who)
        return self._researcher(tier, role, task, messages, known, tools, who)

    @staticmethod
    def _past_queries(messages: list[dict[str, Any]], tool: str) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            for tc in m.get("tool_calls") or []:
                if tc.get("function", {}).get("name") == tool:
                    try:
                        out.append(json.loads(tc["function"].get("arguments") or "{}"))
                    except ValueError:
                        out.append({})
        return out

    def _entities(self, task: Task, known: Known) -> tuple[list[str], str | None]:
        """Entities to cover, plus a prerequisite search if they are not known yet."""
        if task.kind == "multihop":
            owner = known.features.get((task.feature or "").lower())
            return ([owner[0]] if owner else []), (None if owner else "feature")
        if task.kind == "aggregate":
            return (known.catalog or []), (None if known.catalog else "catalog")
        return task.entities, None

    def _researcher(self, tier: str, role: str, task: Task, messages: list[dict[str, Any]],
                    known: Known, tools: set[str], who: tuple[Any, ...]
                    ) -> tuple[str | None, list[tuple[str, dict[str, Any]]]]:
        entities, prereq = self._entities(task, known)
        past = [q.get("query", "") for q in self._past_queries(messages, "search_knowledge_base")]
        if "search_knowledge_base" in tools:
            if prereq == "feature" and past.count(f"which product introduced {task.feature}") < 2:
                return None, [("search_knowledge_base",
                               {"query": f"which product introduced {task.feature}"})]
            if prereq == "catalog" and past.count("Contoso product catalog") < 2:
                return None, [("search_knowledge_base", {"query": "Contoso product catalog"})]
            queries = []
            for e in entities:
                missing = [a for a in task.attrs if a not in known.facts.get(e, {})]
                if not missing:
                    continue
                attempts = sum(1 for q in past if e in q)
                if attempts >= self.cfg.max_query_attempts:
                    continue
                queries.append(self._query(e, missing, attempts))
                if len(queries) >= self.cfg.parallel_searches:
                    break
            if queries:
                return None, [("search_knowledge_base", {"query": q}) for q in queries]
        return self._answer(tier, role, task, entities, known, who), []

    @staticmethod
    def _query(entity: str, attrs: list[str], attempt: int) -> str:
        labels = " ".join(ATTRIBUTES[a] for a in attrs)
        if attempt == 0:
            return f"{entity} {labels}"
        if attempt == 1:
            hints = " ".join(dict.fromkeys(DOC_HINT[a] for a in attrs))
            return f"{entity} {hints}"
        return f"{labels} of {entity}"

    def _value(self, tier: str, known: Known, e: str, a: str, who: tuple[Any, ...]
               ) -> tuple[str, str] | None:
        if a in known.facts.get(e, {}):
            return known.facts[e][a]
        if self._roll(*who, "hallucinate", e, a) < self.cfg.hallucination_rate[tier]:
            return GUESS[a], "none"
        return None

    def _answer(self, tier: str, role: str, task: Task, entities: list[str], known: Known,
                who: tuple[Any, ...]) -> str:
        if role == "worker":
            lines = []
            if task.kind == "multihop":
                owner = known.features.get((task.feature or "").lower())
                if owner:
                    lines.append(f"FINDING | {owner[0]} | introduced | {task.feature} | {owner[1]}")
            for e in entities:
                for a in task.attrs:
                    v = self._value(tier, known, e, a, who)
                    lines.append(f"FINDING | {e} | {a} | {v[0]} | {v[1]}" if v
                                 else f"NOT FOUND | {e} | {a}")
            return "\n".join(lines) or "NOT FOUND | none | none"
        return self._compose(tier, task, entities, known, who)

    def _compose(self, tier: str, task: Task, entities: list[str], known: Known,
                 who: tuple[Any, ...]) -> str:
        def cite(c: str) -> str:
            return f" [{c}]" if c and c != "none" else ""

        if task.kind == "aggregate":
            matches = []
            for e in entities:
                sla = self._value(tier, known, e, "sla", who)
                reg = self._value(tier, known, e, "regions", who)
                if sla and reg and sla[0] == task.sla and task.region and task.region in reg[0]:
                    matches.append(f"{e}{cite(sla[1])}")
            if not entities:
                return "I could not find the product catalog in the knowledge base."
            return ("Products that match: " + ", ".join(matches) + ".") if matches else \
                "No products match both conditions."
        parts = []
        if task.kind == "multihop":
            owner = known.features.get((task.feature or "").lower())
            if not owner:
                return f"I could not find which product introduced {task.feature}."
            parts.append(f"The product that introduced {task.feature} is {owner[0]}"
                         f"{cite(owner[1])}.")
        for e in entities:
            items = []
            for a in task.attrs:
                v = self._value(tier, known, e, a, who)
                items.append(f"{ATTRIBUTES[a]} {v[0]}{cite(v[1])}" if v
                             else f"{ATTRIBUTES[a]} not found in the knowledge base")
            parts.append(f"{e}: " + "; ".join(items) + ".")
        return "\n".join(parts) if parts else "I could not find the products in the question."

    # ------------------------------------------------------------------ lead

    def _lead(self, tier: str, task: Task, question: str, system: str,
              messages: list[dict[str, Any]], known: Known, tools: set[str],
              who: tuple[Any, ...]) -> tuple[str | None, list[tuple[str, dict[str, Any]]]]:
        m = re.search(r"up to (\d+) subtasks", system)
        max_workers = int(m.group(1)) if m else 3
        waves = sum(1 for msg in messages if any(
            tc.get("function", {}).get("name") == "research_subtask"
            for tc in msg.get("tool_calls") or []))
        entities, prereq = self._entities(task, known)
        can_delegate = "research_subtask" in tools and waves <= self.cfg.lead_followup_waves
        if task.kind == "multihop" and prereq == "feature" and can_delegate:
            return None, [("research_subtask", {"instruction": question if waves == 0 else
                                                question + " Check the release notes."})]
        if prereq == "catalog" and "search_knowledge_base" in tools:
            past = self._past_queries(messages, "search_knowledge_base")
            if len(past) < 2:
                return None, [("search_knowledge_base", {"query": "Contoso product catalog"})]
        missing = {e: [a for a in task.attrs if a not in known.facts.get(e, {})]
                   for e in entities}
        missing = {e: attrs for e, attrs in missing.items() if attrs}
        if can_delegate and missing:
            todo = list(missing)
            groups = [todo[i::max_workers] for i in range(min(max_workers, len(todo)))]
            calls = []
            for g in groups:
                attrs = sorted({a for e in g for a in missing[e]}, key=task.attrs.index)
                labels = " and ".join(ATTRIBUTES[a] for a in attrs)
                note = "" if waves == 0 else " A first pass missed these; try other pages too."
                calls.append(("research_subtask",
                              {"instruction": f"Find the {labels} for {', '.join(g)}.{note}"}))
            return None, calls
        return self._compose(tier, task, entities, known, who), []


def mock_transport(sim: SimLLM) -> httpx.MockTransport:
    return httpx.MockTransport(sim.handle)
