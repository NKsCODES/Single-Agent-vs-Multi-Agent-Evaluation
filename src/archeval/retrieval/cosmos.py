"""Azure Cosmos DB for NoSQL as the vector and full-text store.

The container stores each chunk with its embedding. Retrieval is one hybrid query:

    SELECT TOP 3 c.id, c.title, c.text FROM c
    ORDER BY RANK RRF(VectorDistance(c.embedding, @embedding), FullTextScore(c.text, @t0, @t1))

which fuses vector similarity with BM25 keyword scoring using Reciprocal Rank Fusion. The
account needs the NoSQL vector search and full-text search capabilities enabled (the Bicep
template in ``infra/`` turns both on). Authentication uses Entra ID through
``DefaultAzureCredential`` (managed identity in Azure), so no key is stored anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from archeval.corpus.contoso import Document
from archeval.retrieval.embeddings import Embedder
from archeval.retrieval.store import SearchHit, keywords

MAX_TERMS = 8


def vector_embedding_policy(dimensions: int) -> dict[str, Any]:
    return {"vectorEmbeddings": [{"path": "/embedding", "dataType": "float32",
                                  "distanceFunction": "cosine", "dimensions": dimensions}]}


def full_text_policy() -> dict[str, Any]:
    return {"defaultLanguage": "en-US",
            "fullTextPaths": [{"path": "/text", "language": "en-US"}]}


def indexing_policy() -> dict[str, Any]:
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": "/embedding/*"}, {"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
        "fullTextIndexes": [{"path": "/text"}],
    }


def hybrid_query(top_k: int, terms: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """Build the parameterised hybrid query. Terms are always bound as parameters."""
    k = max(1, min(int(top_k), 50))
    terms = list(terms)[:MAX_TERMS] or ["contoso"]
    names = [f"@t{i}" for i in range(len(terms))]
    query = (f"SELECT TOP {k} c.id, c.title, c.text FROM c ORDER BY RANK RRF("
             f"VectorDistance(c.embedding, @embedding), FullTextScore(c.text, {', '.join(names)}))")
    params: list[dict[str, Any]] = [{"name": n, "value": t} for n, t in zip(names, terms,
                                                                             strict=True)]
    return query, params


class CosmosHybridStore:
    def __init__(self, embedder: Embedder, endpoint: str, database: str, container: str,
                 key: str | None = None, client: Any = None) -> None:
        self.embedder = embedder
        self.database_name, self.container_name = database, container
        if client is None:  # pragma: no cover - needs an Azure resource
            from azure.cosmos.aio import CosmosClient

            if key:
                client = CosmosClient(endpoint, credential=key)
            else:
                from azure.identity.aio import DefaultAzureCredential

                client = CosmosClient(endpoint, credential=DefaultAzureCredential())
        self.client = client
        self._container: Any = None

    async def container(self) -> Any:
        if self._container is None:
            from azure.cosmos import PartitionKey

            db = await self.client.create_database_if_not_exists(self.database_name)
            self._container = await db.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/doc_id"),
                indexing_policy=indexing_policy(),
                vector_embedding_policy=vector_embedding_policy(self.embedder.dimensions),
                full_text_policy=full_text_policy(),
            )
        return self._container

    async def upsert(self, docs: Sequence[Document]) -> int:
        container = await self.container()
        vectors = await self.embedder.embed([d.text for d in docs])
        for d, v in zip(docs, vectors, strict=True):
            await container.upsert_item({"id": d.id, "doc_id": d.id, "title": d.title,
                                         "text": d.text, "product": d.product, "embedding": v})
        return len(docs)

    async def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        container = await self.container()
        embedding = (await self.embedder.embed([query]))[0]
        sql, params = hybrid_query(top_k, keywords(query))
        params.append({"name": "@embedding", "value": embedding})
        hits: list[SearchHit] = []
        async for item in container.query_items(query=sql, parameters=params):
            hits.append(SearchHit(item["id"], item.get("title", ""), item["text"],
                                  float(len(hits))))
        return hits

    async def ping(self) -> bool:
        await self.container()
        return True
