"""Embedding models.

``AzureOpenAIEmbedder`` calls an Azure OpenAI embedding deployment with Entra ID (keyless) or a
key. ``HashingEmbedder`` is a deterministic, offline feature-hashing embedder used in tests and
the offline benchmark, so no network or model download is needed.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = Counter(toks + [f"{a}_{b}" for a, b in pairwise(toks)])
        vec = np.zeros(self.dimensions, dtype=np.float32)
        for feat, tf in feats.items():
            h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimensions
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(tf))
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm else vec.tolist()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class AzureOpenAIEmbedder:  # pragma: no cover - needs an Azure resource
    def __init__(self, endpoint: str, deployment: str, api_version: str,
                 api_key: str | None = None, dimensions: int = 1536) -> None:
        from openai import AsyncAzureOpenAI

        kwargs: dict[str, Any] = {"azure_endpoint": endpoint, "api_version": api_version}
        if api_key:
            kwargs["api_key"] = api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        self.client = AsyncAzureOpenAI(**kwargs)
        self.deployment = deployment
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = await self.client.embeddings.create(model=self.deployment, input=list(texts),
                                                  dimensions=self.dimensions)
        return [d.embedding for d in out.data]
