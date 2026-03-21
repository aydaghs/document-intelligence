from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_LOG = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = False


class SemanticSearch:
    """Semantic search using sentence-transformers when available, TF-IDF otherwise."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._use_transformers = False
        self._model = None
        self._tfidf: Optional[TfidfVectorizer] = None

        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(model_name)
                self._use_transformers = True
                _LOG.info("SemanticSearch: using sentence-transformers (%s)", model_name)
            except Exception as exc:
                _LOG.warning("sentence-transformers load failed (%s); falling back to TF-IDF", exc)

        if not self._use_transformers:
            _LOG.info("SemanticSearch: using TF-IDF fallback (scikit-learn)")

    @property
    def backend(self) -> str:
        return "sentence-transformers" if self._use_transformers else "tfidf"

    def embed(self, texts: List[str]) -> np.ndarray:
        if self._use_transformers and self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # TF-IDF fallback: fit on these texts + return dense vectors
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
        mat = vec.fit_transform(texts).toarray().astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).reshape(-1)

    def search(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if self._use_transformers and self._model is not None:
            q_emb = self._model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            # TF-IDF: fit on all candidate texts + query together so vocabulary matches
            corpus = [c.get("text", "") for c in candidates] + [query_text]
            vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
            mat = vec.fit_transform(corpus).toarray().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            candidate_embeddings = mat[:-1]
            q_emb = mat[-1]

        scores = self.similarity(q_emb, candidate_embeddings)
        ranked = sorted(enumerate(scores), key=lambda iv: iv[1], reverse=True)
        return [{"score": float(scores[i]), **candidates[i]} for i, _ in ranked[:top_k]]

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        return pickle.loads(blob)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        return pickle.dumps(embedding)
