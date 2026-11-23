"""Evidence retrieval: TF-IDF default, true embedding-based retrieval optional.

TF-IDF is a lexical baseline, NOT semantic embeddings. Install project[semantic]
and use --semantic to activate sentence-transformer embeddings.
"""
from __future__ import annotations

import math

from .models import Evidence


class EvidenceRetriever:
    def __init__(self, use_semantic_embeddings: bool = False):
        self.use_semantic_embeddings = use_semantic_embeddings
        self._model = None
        if use_semantic_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install extras: uv sync --extra semantic") from exc
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query: str, items: list[Evidence], k: int = 5) -> list[Evidence]:
        # Ignore stale observations before retrieval; never turn missing evidence into 'safe'.
        eligible = [x for x in items if x.fresh()]
        if not eligible:
            return []
        docs = [f"{x.kind} {x.place} {x.summary} {x.source}" for x in eligible]
        if self._model is not None:
            from sentence_transformers import util
            embeddings = self._model.encode([query, *docs], convert_to_tensor=True)
            sims = util.cos_sim(embeddings[0], embeddings[1:])[0].tolist()
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform([query, *docs])
            sims = cosine_similarity(matrix[0], matrix[1:])[0].tolist()
        ranked = sorted(range(len(eligible)), key=lambda i: (sims[i], eligible[i].official), reverse=True)
        # Only add documents with a nonzero match (unless there is just one observation).
        return [eligible[i] for i in ranked if sims[i] > 0][:k]
