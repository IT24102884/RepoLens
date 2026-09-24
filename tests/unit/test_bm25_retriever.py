import pytest
from ai.core.models import DocumentChunk, DocumentType
from ai.retrieval.bm25_retriever import BM25Retriever, tokenize_code


def test_tokenize_code_handles_identifiers_and_issues():
    text = "AsyncClient hangs on HTTPStatus.TOO_MANY_REQUESTS in issue #1240"
    tokens = tokenize_code(text)
    
    # Check issue tags and plain numbers
    assert "#1240" in tokens
    assert "1240" in tokens
    
    # Check CamelCase splitting
    assert "asyncclient" in tokens
    assert "async" in tokens
    assert "client" in tokens
    
    # Check dotted and snake_case splitting
    assert "httpstatus.too_many_requests" in tokens
    assert "httpstatus" in tokens
    assert "too_many_requests" in tokens
    assert "too" in tokens
    assert "many" in tokens
    assert "requests" in tokens


def test_bm25_retriever_initialization():
    retriever = BM25Retriever()
    assert len(retriever.chunks) > 0
    assert retriever.bm25 is not None


def test_bm25_retrieval_ticket_lookup():
    retriever = BM25Retriever()
    results = retriever.search("Why does client.stream hang according to issue 1240?", top_k=3)
    
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "1240" in top_chunk.file_path
    assert score > 0.0
