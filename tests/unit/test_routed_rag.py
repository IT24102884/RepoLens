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


def test_routed_rag_overview_with_repo_card(mock_routed_rag):
    mock_choice = MagicMock()
    mock_choice.message.content = "HTTPX is a next-generation HTTP client for Python."
    mock_routed_rag.client.chat.completions.create.return_value.choices = [mock_choice]

    mock_routed_rag.router.route = MagicMock(return_value=RouteDecision(
        intent=QueryIntent.DOCS_CONCEPTUAL,
        target_source_type=DocumentType.DOCUMENTATION,
        confidence=0.98,
        route_source="llm_classifier",
        reasoning="User asks for high-level repository overview.",
        optimized_search_query="httpx HTTP client architecture overview README",
    ))

    mock_chunk = DocumentChunk(
        id="chunk_test_1",
        content="HTTPX is a fully featured HTTP client for Python 3.",
        file_path="README.md",
        start_line=1,
        end_line=10,
        source_type=DocumentType.DOCUMENTATION,
    )
    mock_routed_rag.retriever.search = MagicMock(return_value=[(mock_chunk, 0.95)])

    out = mock_routed_rag.answer("give me the big picture of this repo")
    assert out["routing_decision"]["intent"] == QueryIntent.DOCS_CONCEPTUAL
    assert out["answer"] == "HTTPX is a next-generation HTTP client for Python."

    mock_routed_rag.retriever.search.assert_called_with(
        query="httpx HTTP client architecture overview README",
        top_k=mock_routed_rag.top_k,
        filter_type=DocumentType.DOCUMENTATION,
    )

    call_args = mock_routed_rag.client.chat.completions.create.call_args[1]["messages"]
    system_instruction = call_args[0]["content"]
    assert "=== REPOSITORY IDENTITY CARD ===" in system_instruction
    assert "Repository: encode/httpx" in system_instruction
    assert "README.md" in system_instruction

