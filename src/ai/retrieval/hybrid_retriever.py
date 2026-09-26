import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.core.models import DocumentChunk, DocumentType
from ai.retrieval.bm25_retriever import BM25Retriever
from ai.retrieval.chroma_retriever import ChromaRetriever

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


class HybridRetriever:
    """Hybrid Retriever combining ChromaDB Dense Vector Search and BM25 Sparse Search
    using Reciprocal Rank Fusion (RRF with k=60).
    
    Includes Mathematical Distance Gating: Drops candidate union if the top dense match
    falls below the noise floor (default 0.28 cosine similarity).
    """

    def __init__(
        self,
        dense_retriever: Optional[ChromaRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
        default_min_similarity: float = 0.28,
    ):
        self.dense_retriever = dense_retriever or ChromaRetriever()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.rrf_k = rrf_k
        self.default_min_similarity = default_min_similarity

    @traceable(name="Hybrid RRF Search", run_type="retriever")
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[DocumentType] = None,
        candidate_pool_size: int = 50,
        min_dense_similarity: Optional[float] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid search with Reciprocal Rank Fusion and mathematical distance gating."""
        threshold = min_dense_similarity if min_dense_similarity is not None else self.default_min_similarity

        # 1. Fetch dense candidates
        dense_results: List[Tuple[DocumentChunk, float]] = self.dense_retriever.search(
            query=query,
            top_k=candidate_pool_size,
            filter_type=filter_type,
        )

        # 2. Fetch sparse candidates
        sparse_results: List[Tuple[DocumentChunk, float]] = self.sparse_retriever.search(
            query=query,
            top_k=candidate_pool_size,
            filter_type=filter_type,
        )

        # Mathematical Distance Gate: Only drop when dense similarity is below noise floor AND no sparse keyword matches
        if threshold > 0.0:
            top_sim = dense_results[0][1] if dense_results else 0.0
            if top_sim < threshold and not sparse_results:
                return []

        chunk_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # 3. Accumulate dense ranks
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            chunk_map[chunk.id] = chunk
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + score

        # 4. Accumulate sparse ranks
        for rank, (chunk, _) in enumerate(sparse_results, start=1):
            chunk_map[chunk.id] = chunk
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + score

        # 5. Sort candidate union by fused RRF score descending
        fused_ranking = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        output: List[Tuple[DocumentChunk, float]] = []
        for chunk_id, fused_score in fused_ranking[:top_k]:
            output.append((chunk_map[chunk_id], fused_score))

        return output
