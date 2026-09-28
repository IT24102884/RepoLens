from unittest.mock import MagicMock
import pytest

from ai.agents.critic_rag import CriticRAG
from ai.core.models import DocumentChunk, DocumentType
from ai.critic.citation_verifier import CitationStatus
from ai.critic.critic_agent import CriticResult, CriticStatus
from ai.routing.router import QueryIntent, RouteDecision


@pytest.fixture
def sample_chunk():
    return DocumentChunk(
        id="chunk_url_1",
        file_path="httpx/_urls.py",
        start_line=10,
        end_line=50,
        content="class URL:\n    def __init__(self, url): ...",
        source_type=DocumentType.CODE,
    )


@pytest.fixture
def mock_critic_rag(sample_chunk):
    routed_mock = MagicMock()
    critic_mock = MagicMock()

    rag = CriticRAG(
        routed_rag=routed_mock,
        critic_agent=critic_mock,
    )
    return rag, routed_mock, critic_mock


def test_critic_rag_verified_flow(mock_critic_rag, sample_chunk):
    rag, routed_mock, critic_mock = mock_critic_rag

    routed_mock.answer.return_value = {
        "query": "Where is URL class?",
        "answer": "The URL class is declared in [httpx/_urls.py L15-L35].",
        "retrieved_chunks": [sample_chunk.model_dump()],
        "retrieval_latency_ms": 150.0,
        "generation_latency_ms": 800.0,
        "total_latency_ms": 950.0,
        "system_version": "System C (Cascading Intent Router)",
        "routing_decision": {"intent": QueryIntent.CODE_SYMBOL.value},
    }

    critic_mock.critique.return_value = CriticResult(
        status=CriticStatus.VERIFIED,
        faithfulness_score=1.0,
        hallucinations_detected=[],
        verified_answer="The URL class is declared in [httpx/_urls.py L15-L35].",
        critique_summary="All claims are verified.",
        latency_ms=250.0,
    )

    out = rag.answer("Where is URL class?")

    assert out["critic_status"] == CriticStatus.VERIFIED.value
    assert out["faithfulness_score"] == 1.0
    assert out["system_version"] == "System D (Self-Correction Critic & Citation Verifier)"
    assert out["critic_latency_ms"] == 250.0
    assert out["total_latency_ms"] == 1200.0
    assert len(out["hallucinations_detected"]) == 0


def test_critic_rag_drifted_citation_repair_and_pruning(mock_critic_rag, sample_chunk):
    rag, routed_mock, critic_mock = mock_critic_rag

    # LLM cited L200-L240, which drifted from real chunk L10-L50
    draft_with_drift = "The URL class is in [httpx/_urls.py L200-L240] and caches to Redis."
    routed_mock.answer.return_value = {
        "query": "Explain URL",
        "answer": draft_with_drift,
        "retrieved_chunks": [sample_chunk.model_dump()],
        "retrieval_latency_ms": 120.0,
        "generation_latency_ms": 700.0,
        "total_latency_ms": 820.0,
        "system_version": "System C (Cascading Intent Router)",
        "routing_decision": {"intent": QueryIntent.CODE_SYMBOL.value},
    }

    # Critic prunes Redis claim and verifies answer
    critic_mock.critique.return_value = CriticResult(
        status=CriticStatus.REVISED,
        faithfulness_score=0.92,
        hallucinations_detected=["Removed ungrounded Redis caching claim"],
        verified_answer="The URL class is in [httpx/_urls.py L10-L50].",
        critique_summary="Pruned Redis extrapolation.",
        latency_ms=300.0,
    )

    out = rag.answer("Explain URL")

    assert out["critic_status"] == CriticStatus.REVISED.value
    assert out["faithfulness_score"] == 0.92
    assert "Redis" in out["hallucinations_detected"][0]
    assert "Repaired 1 drifted line coordinate(s)" in out["critique_summary"]
    # Verify the input to the critic was already line-repaired
    called_draft = critic_mock.critique.call_args[1]["draft_answer"]
    assert "[httpx/_urls.py L10-L50]" in called_draft


def test_critic_rag_out_of_scope_instant_bypass(mock_critic_rag):
    rag, routed_mock, critic_mock = mock_critic_rag

    routed_mock.answer.return_value = {
        "query": "weather forecast in Tokyo",
        "answer": "This query is out of scope.",
        "retrieved_chunks": [],
        "retrieval_latency_ms": 0.0,
        "generation_latency_ms": 0.0,
        "total_latency_ms": 0.5,
        "system_version": "System C (Cascading Intent Router)",
        "routing_decision": {"intent": QueryIntent.OUT_OF_SCOPE.value},
    }

    out = rag.answer("weather forecast in Tokyo")

    assert out["critic_status"] == CriticStatus.VERIFIED.value
    assert out["critic_latency_ms"] == 0.0
    # Critic agent shouldn't even be invoked for out-of-scope refusals
    critic_mock.critique.assert_not_called()
