from unittest.mock import MagicMock, patch
import pytest

from ai.agents.routed_rag import RoutedRAG
from ai.core.models import DocumentChunk, DocumentType
from ai.routing.router import QueryIntent, RouteDecision


@pytest.fixture
def mock_routed_rag():
    with patch("ai.agents.routed_rag.Groq"):
        rag = RoutedRAG()
        rag.client = MagicMock()
        return rag


def test_routed_rag_out_of_scope_instant_refusal(mock_routed_rag):
    out = mock_routed_rag.answer("How do I configure GraphQL with Apollo Federation in HTTPX?")
    assert len(out["retrieved_chunks"]) == 0
    assert "outside the scope" in out["answer"]
    assert out["retrieval_latency_ms"] == 0.0
    assert out["generation_latency_ms"] == 0.0
    assert out["routing_decision"]["intent"] == QueryIntent.OUT_OF_SCOPE


def test_routed_rag_code_symbol_filtered(mock_routed_rag):
    mock_choice = MagicMock()
    mock_choice.message.content = "The URL class is defined in `httpx/_urls.py`."
    mock_routed_rag.client.chat.completions.create.return_value.choices = [mock_choice]

    out = mock_routed_rag.answer("Where is the URL class defined and how does it parse raw byte paths?")
    assert out["routing_decision"]["intent"] == QueryIntent.CODE_SYMBOL
    assert len(out["retrieved_chunks"]) > 0
    for c in out["retrieved_chunks"]:
        assert c["source_type"] == DocumentType.CODE
