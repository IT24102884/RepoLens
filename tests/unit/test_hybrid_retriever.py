import pytest
from ai.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_initialization():
    hybrid = HybridRetriever()
    assert hybrid.dense_retriever is not None
    assert hybrid.sparse_retriever is not None
    assert hybrid.rrf_k == 60


def test_hybrid_retriever_ticket_recall():
    hybrid = HybridRetriever()
    results = hybrid.search("How was the HTTP/2 keepalive connection drop issue fixed in ticket 1405?", top_k=3)
    
    assert len(results) > 0
    top_chunk, rrf_score = results[0]
    assert "1405" in top_chunk.file_path
    assert rrf_score > 0.0


def test_hybrid_retriever_rrf_scoring_order():
    hybrid = HybridRetriever()
    results = hybrid.search("What are the timeout configuration parameters?", top_k=5)
    
    assert len(results) > 0
    # Verify scores are strictly monotonically descending or equal
    scores = [s for _, s in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
