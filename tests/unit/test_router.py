from unittest.mock import MagicMock
import pytest

from ai.core.models import DocumentType
from ai.routing.router import IntentRouter, QueryIntent, RouteDecision


@pytest.fixture
def router():
    return IntentRouter(api_key="mock_key")


def test_heuristic_out_of_scope(router):
    test_queries = [
        "How do I configure a GraphQL subscription with Apollo Federation in HTTPX?",
        "What is the weather in Tokyo tomorrow?",
        "Give me a recipe for bake a chocolate cake",
        "What is the current bitcoin price?",
    ]
    for q in test_queries:
        decision = router.classify_heuristically(q)
        assert decision is not None
        assert decision.intent == QueryIntent.OUT_OF_SCOPE
        assert decision.confidence >= 0.90
        assert decision.route_source == "heuristic"


def test_heuristic_bug_ticket(router):
    test_queries = [
        "Why does AsyncClient hang when calling client.stream according to issue 1240?",
        "How was the connection drop fixed in ticket 1405?",
        "Details on PR #45 regression",
    ]
    for q in test_queries:
        decision = router.classify_heuristically(q)
        assert decision is not None
        assert decision.intent == QueryIntent.BUG_TICKET
        assert decision.target_source_type == DocumentType.ISSUE_PR
        assert decision.confidence >= 0.90
        assert decision.route_source == "heuristic"


def test_heuristic_code_symbol(router):
    test_queries = [
        "Where is the URL class defined and how does it parse raw byte paths?",
        "Which file and class defines the codes status HTTPStatus.TOO_MANY_REQUESTS handling?",
        "def _send_handling_redirects implementation in _client.py",
    ]
    for q in test_queries:
        decision = router.classify_heuristically(q)
        assert decision is not None
        assert decision.intent == QueryIntent.CODE_SYMBOL
        assert decision.target_source_type == DocumentType.CODE
        assert decision.confidence >= 0.90


def test_heuristic_docs_conceptual(router):
    test_queries = [
        "What are the four different types of timeouts supported in HTTPX and what exceptions do they raise?",
        "How do I configure custom SSL certificates or disable SSL verification in an HTTPX client?",
    ]
    for q in test_queries:
        decision = router.classify_heuristically(q)
        assert decision is not None
        assert decision.intent == QueryIntent.DOCS_CONCEPTUAL
        assert decision.target_source_type == DocumentType.DOCUMENTATION
        assert decision.confidence >= 0.90


def test_heuristic_multi_hop(router):
    q = "How does Client.request pass headers and cookies down to the underlying transport dispatch?"
    decision = router.classify_heuristically(q)
    assert decision is not None
    assert decision.intent == QueryIntent.MULTI_HOP
    assert decision.target_source_type == DocumentType.CODE
    assert len(decision.sub_queries) >= 2


def test_llm_classification_success(router):
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '```json\n'
        '{\n'
        '  "intent": "DOCS_CONCEPTUAL",\n'
        '  "confidence": 0.96,\n'
        '  "reasoning": "User asks for general architectural principles of connection pooling.",\n'
        '  "sub_queries": []\n'
        '}\n'
        '```'
    )
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]
    router._groq_client = mock_groq

    decision = router.classify_with_llm("Explain the connection pool life cycle")
    assert decision.intent == QueryIntent.DOCS_CONCEPTUAL
    assert decision.target_source_type == DocumentType.DOCUMENTATION
    assert decision.confidence == 0.96
    assert decision.route_source == "llm_classifier"


def test_llm_classification_fallback_on_error(router):
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = RuntimeError("Groq connection timeout")
    router._groq_client = mock_groq

    decision = router.classify_with_llm("Some random ambiguous query")
    assert decision.intent == QueryIntent.CODE_SYMBOL
    assert decision.confidence == 0.5
    assert decision.route_source == "llm_classifier_fallback"
