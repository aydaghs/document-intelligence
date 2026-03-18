from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from sklearn.metrics.pairwise import cosine_similarity

_LOG = logging.getLogger(__name__)


class SemanticSearch:
    """Semantic search support using sentence-transformers embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it to enable semantic search."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        # cosine_similarity expects 2D arrays
        return cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).reshape(-1)

    def search(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-K most similar candidate dicts (expects candidate_embeddings aligned)."""

        if len(candidates) == 0:
            return []

        q_emb = self.embed([query_text])[0]
        scores = self.similarity(q_emb, candidate_embeddings)
        ranked = sorted(enumerate(scores), key=lambda iv: iv[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx, score in ranked[:top_k]:
            results.append({"score": float(score), **candidates[idx]})
        return results

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        return pickle.loads(blob)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        return pickle.dumps(embedding)
