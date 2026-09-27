from unittest.mock import MagicMock
import pytest

from ai.core.models import DocumentChunk, DocumentType
from ai.critic.critic_agent import CriticAgent, CriticResult, CriticStatus


@pytest.fixture
def mock_chunks():
    return [
        DocumentChunk(
            id="chunk_1",
            file_path="httpx-0.27.0/httpx/_urls.py",
            start_line=10,
            end_line=50,
            content="class URL:\n    def __init__(self, url): self._uri = url",
            source_type=DocumentType.CODE,
        )
    ]


@pytest.fixture
def critic():
    return CriticAgent(api_key="mock_key")


def test_critic_agent_verified_pass(critic, mock_chunks):
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{\n'
        '  "status": "VERIFIED",\n'
        '  "faithfulness_score": 1.0,\n'
        '  "hallucinations_detected": [],\n'
        '  "verified_answer": "The URL class is defined in httpx/_urls.py.",\n'
        '  "critique_summary": "All statements are directly entailed by the context."\n'
        '}'
    )
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]
    critic._groq_client = mock_groq

    result = critic.critique(
        query="Where is URL defined?",
        draft_answer="The URL class is defined in httpx/_urls.py.",
        context_chunks=mock_chunks,
    )

    assert result.status == CriticStatus.VERIFIED
    assert result.faithfulness_score == 1.0
    assert len(result.hallucinations_detected) == 0
    assert "The URL class is defined" in result.verified_answer


def test_critic_agent_prunes_hallucination(critic, mock_chunks):
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{\n'
        '  "status": "REVISED",\n'
        '  "faithfulness_score": 0.90,\n'
        '  "hallucinations_detected": ["Extrapolated native Redis cache support"],\n'
        '  "verified_answer": "The URL class parses URLs. It does not provide Redis caching.",\n'
        '  "critique_summary": "Pruned ungrounded claim regarding Redis support."\n'
        '}'
    )
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]
    critic._groq_client = mock_groq

    draft = "The URL class parses URLs and provides automatic Redis caching."
    result = critic.critique(
        query="Tell me about URL caching",
        draft_answer=draft,
        context_chunks=mock_chunks,
    )

    assert result.status == CriticStatus.REVISED
    assert result.faithfulness_score == 0.90
    assert "Redis" in result.hallucinations_detected[0]
    assert result.verified_answer != draft
    assert "does not provide Redis" in result.verified_answer


def test_critic_agent_insufficient_evidence_when_no_chunks(critic):
    result = critic.critique(
        query="Does HTTPX support WebSockets?",
        draft_answer="HTTPX does not support WebSockets out of the box.",
        context_chunks=[],
    )

    assert result.status == CriticStatus.INSUFFICIENT_EVIDENCE
    assert result.faithfulness_score == 0.0
    assert "does not contain verified documentation" in result.verified_answer


def test_critic_agent_fallback_on_error(critic, mock_chunks):
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = RuntimeError("Groq rate limit exceeded")
    critic._groq_client = mock_groq

    draft = "Some draft response."
    result = critic.critique(
        query="Any query",
        draft_answer=draft,
        context_chunks=mock_chunks,
    )

    # Should fall back gracefully to returning draft without crashing
    assert result.verified_answer == draft
    assert "fallback due to error" in result.critique_summary
