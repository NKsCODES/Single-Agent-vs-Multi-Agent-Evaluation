"""Hybrid retrieval: vector similarity fused with BM25 by Reciprocal Rank Fusion (RRF).

``InMemoryHybridStore`` implements the same ranking that Azure Cosmos DB for NoSQL runs for
``ORDER BY RANK RRF(VectorDistance(...), FullTextScore(...))``, so the offline benchmark and
the Cosmos DB deployment rank results the same way.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from archeval.corpus.contoso import Document
from archeval.retrieval.embeddings import Embedder, tokenize

RRF_K = 60  # the constant from the original RRF paper, also the common default

STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "is", "what",
             "which", "with", "by", "per", "are", "does", "do", "its", "it", "that", "this"}


@dataclass(frozen=True)
class SearchHit:
    id: str
    title: str
    text: str
    score: float


class HybridStore(Protocol):
    async def upsert(self, docs: Sequence[Document]) -> int: ...
    async def search(self, query: str, top_k: int = 3) -> list[SearchHit]: ...
    async def ping(self) -> bool: ...


def keywords(query: str) -> list[str]:
    return [t for t in tokenize(query) if t not in STOPWORDS]


def rrf(rankings: Sequence[Sequence[str]], weights: Sequence[float] | None = None,
        k: int = RRF_K) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class InMemoryHybridStore:
    def __init__(self, embedder: Embedder, k1: float = 1.2, b: float = 0.75) -> None:
        self.embedder = embedder
        self.k1, self.b = k1, b
        self.docs: list[Document] = []
        self.matrix: np.ndarray = np.zeros((0, embedder.dimensions), dtype=np.float32)
        self.tfs: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        self.avg_len = 0.0

    async def upsert(self, docs: Sequence[Document]) -> int:
        by_id = {d.id: d for d in self.docs}
        by_id.update({d.id: d for d in docs})
        self.docs = sorted(by_id.values(), key=lambda d: d.id)
        vectors = await self.embedder.embed([d.text for d in self.docs])
        self.matrix = np.asarray(vectors, dtype=np.float32)
        self.tfs = [Counter(tokenize(d.text)) for d in self.docs]
        self.df = Counter(t for tf in self.tfs for t in tf)
        self.avg_len = sum(sum(tf.values()) for tf in self.tfs) / max(1, len(self.tfs))
        return len(docs)

    def _bm25(self, terms: Sequence[str]) -> np.ndarray:
        n = len(self.docs)
        scores = np.zeros(n, dtype=np.float64)
        for t in set(terms):
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    dl = sum(tf.values())
                    scores[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
        return scores

    async def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        if not self.docs:
            return []
        qv = np.asarray((await self.embedder.embed([query]))[0], dtype=np.float32)
        cos = self.matrix @ qv
        bm = self._bm25(keywords(query))
        ids = [d.id for d in self.docs]
        vec_rank = [ids[i] for i in np.argsort(-cos, kind="stable")]
        txt_rank = [ids[i] for i in np.argsort(-bm, kind="stable") if bm[i] > 0]
        fused = rrf([vec_rank, txt_rank])[:top_k]
        by_id = {d.id: d for d in self.docs}
        return [SearchHit(i, by_id[i].title, by_id[i].text, round(s, 6)) for i, s in fused]

    async def ping(self) -> bool:
        return True
