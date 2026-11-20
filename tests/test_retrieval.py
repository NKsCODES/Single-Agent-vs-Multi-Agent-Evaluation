from __future__ import annotations

from typing import Any

from archeval.corpus.contoso import DOCUMENTS, FACTS
from archeval.retrieval.cosmos import (
    CosmosHybridStore,
    full_text_policy,
    hybrid_query,
    indexing_policy,
    vector_embedding_policy,
)
from archeval.retrieval.embeddings import HashingEmbedder
from archeval.retrieval.store import rrf


async def test_hybrid_search_finds_the_product_overview(store) -> None:  # type: ignore[no-untyped-def]
    hits = await store.search("Contoso Vault uptime SLA", 3)
    assert hits[0].id == "vault-overview"


async def test_feature_search_finds_release_notes(store) -> None:  # type: ignore[no-untyped-def]
    feature = FACTS["Contoso Chat"].releases[0][2]
    hits = await store.search(f"which product introduced {feature}", 3)
    assert "chat-releases" in [h.id for h in hits]


def test_rrf_rewards_agreement() -> None:
    fused = [doc for doc, _ in rrf([["a", "b", "c"], ["b", "a", "d"]])]
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("d") > fused.index("a")


async def test_hashing_embedder_is_normalised_and_deterministic() -> None:
    e = HashingEmbedder(64)
    v1, v2 = await e.embed(["Contoso Mail SLA", "Contoso Mail SLA"])
    assert v1 == v2 and abs(sum(x * x for x in v1) - 1) < 1e-6


def test_cosmos_policies_enable_vector_and_full_text_search() -> None:
    vp = vector_embedding_policy(384)["vectorEmbeddings"][0]
    assert vp == {"path": "/embedding", "dataType": "float32", "distanceFunction": "cosine",
                  "dimensions": 384}
    ip = indexing_policy()
    assert ip["vectorIndexes"][0]["type"] == "diskANN"
    assert ip["fullTextIndexes"][0]["path"] == "/text"
    assert {"path": "/embedding/*"} in ip["excludedPaths"]
    assert full_text_policy()["fullTextPaths"][0]["path"] == "/text"


def test_hybrid_query_uses_rrf_of_vector_and_full_text() -> None:
    sql, params = hybrid_query(3, ["vault", "sla"])
    assert sql.startswith("SELECT TOP 3 c.id, c.title, c.text FROM c ORDER BY RANK RRF(")
    assert "VectorDistance(c.embedding, @embedding)" in sql
    assert "FullTextScore(c.text, @t0, @t1)" in sql
    assert {"name": "@t0", "value": "vault"} in params


class FakeContainer:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.queries: list[tuple[str, Any]] = []

    async def upsert_item(self, item: dict[str, Any]) -> None:
        self.items.append(item)

    def query_items(self, query: str, parameters: Any) -> Any:
        self.queries.append((query, parameters))
        items = self.items

        async def gen():  # type: ignore[no-untyped-def]
            for it in items[:3]:
                yield {"id": it["id"], "title": it["title"], "text": it["text"]}

        return gen()

    async def read(self) -> dict[str, str]:
        return {"id": "chunks"}


class FakeDatabase:
    def __init__(self, container: FakeContainer) -> None:
        self.container, self.kwargs = container, {}

    async def create_container_if_not_exists(self, **kwargs: Any) -> FakeContainer:
        self.kwargs = kwargs
        return self.container


class FakeCosmos:
    def __init__(self) -> None:
        self.db = FakeDatabase(FakeContainer())

    async def create_database_if_not_exists(self, name: str) -> FakeDatabase:
        return self.db


async def test_cosmos_store_creates_container_with_policies_and_queries_hybrid() -> None:
    fake = FakeCosmos()
    store = CosmosHybridStore(HashingEmbedder(32), "https://x", "db", "chunks", client=fake)
    assert await store.upsert(DOCUMENTS[:5]) == 5
    item = fake.db.container.items[0]
    assert len(item["embedding"]) == 32 and item["doc_id"] == item["id"]
    assert fake.db.kwargs["vector_embedding_policy"]["vectorEmbeddings"][0]["dimensions"] == 32
    hits = await store.search("Contoso Mail uptime SLA", 3)
    assert len(hits) == 3
    sql, params = fake.db.container.queries[0]
    assert "ORDER BY RANK RRF" in sql and params[-1]["name"] == "@embedding"
    assert await store.ping()
