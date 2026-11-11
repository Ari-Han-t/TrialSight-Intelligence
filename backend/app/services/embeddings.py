"""Embedding providers for the dense half of hybrid retrieval.

- ``hash``    : deterministic feature-hashing of tokens (offline, zero-cost fallback; lexical only).
- ``bedrock`` : Amazon Titan Text Embeddings V2 via Amazon Bedrock (true semantic vectors).

Vectors are L2-normalised float32 arrays and are computed once at ingest time and
stored alongside each chunk, so a query costs one embedding call instead of
re-embedding the whole corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from app.core.config import get_settings
from app.rag.chunking import tokenize


logger = logging.getLogger(__name__)
settings = get_settings()


class Embedder:
    name: str = "base"
    dim: int = 0

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0 else (vector / norm).astype(np.float32)


class HashEmbedder(Embedder):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token, count in Counter(tokenize(text)).items():
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[index] += float(count)
        return self._normalise(vector)


class BedrockTitanEmbedder(Embedder):
    def __init__(self, model_id: str, dim: int, region: str | None) -> None:
        import boto3  # imported lazily so the hash path has no AWS dependency at runtime

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.dim = dim
        self.name = f"bedrock:{model_id}:{dim}"

    def embed_query(self, text: str) -> np.ndarray:
        body = json.dumps({"inputText": (text or " ")[:20000], "dimensions": self.dim, "normalize": True})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return self._normalise(np.asarray(payload["embedding"], dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        # Titan V2 embeds one text per call; a small pool keeps ingest fast without tripping throttling.
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(self.embed_query, texts))


def to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


@lru_cache
def get_fallback_embedder() -> Embedder:
    return HashEmbedder(settings.dense_vector_size)


def embed_documents_safe(texts: list[str]) -> tuple[list[np.ndarray], str]:
    """Embed with the configured provider; if it fails (no model access, throttling, network),
    degrade to the hash embedder so ingestion never breaks. Returns (vectors, embedder name)."""
    primary = get_embedder()
    try:
        return primary.embed_documents(texts), primary.name
    except Exception as exc:
        if primary is get_fallback_embedder():
            raise
        logger.warning("Embedding with %s failed (%s); using hash embedder", primary.name, type(exc).__name__)
        fallback = get_fallback_embedder()
        return fallback.embed_documents(texts), fallback.name


def query_vectors(query: str) -> dict[str, np.ndarray]:
    """Query vectors keyed by embedder name: always the hash vector, plus the primary when available."""
    fallback = get_fallback_embedder()
    vectors = {fallback.name: fallback.embed_query(query)}
    primary = get_embedder()
    if primary.name not in vectors:
        try:
            vectors[primary.name] = primary.embed_query(query)
        except Exception as exc:
            logger.warning("Query embedding with %s failed (%s)", primary.name, type(exc).__name__)
    return vectors


@lru_cache
def get_embedder() -> Embedder:
    if settings.embedding_provider == "bedrock":
        try:
            return BedrockTitanEmbedder(
                model_id=settings.bedrock_embedding_model_id,
                dim=settings.bedrock_embedding_dim,
                region=settings.aws_region,
            )
        except Exception:  # pragma: no cover - depends on environment
            logger.exception("Bedrock embedder unavailable; falling back to hashing embedder")
    return get_fallback_embedder()
