import pytest

from ai.core.models import DocumentChunk, DocumentType
from ai.critic.citation_verifier import (
    CitationStatus,
    CitationVerifier,
)


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            id="chunk_urls_1",
            file_path="httpx-0.27.0/httpx/_urls.py",
            start_line=10,
            end_line=50,
            content="class URL:\n    def __init__(self, url): ...",
            source_type=DocumentType.CODE,
        ),
        DocumentChunk(
            id="chunk_client_1",
            file_path="httpx-0.27.0/httpx/_client.py",
            start_line=100,
            end_line=150,
            content="class Client:\n    def request(self, ...): ...",
            source_type=DocumentType.CODE,
        ),
        DocumentChunk(
            id="chunk_readme_1",
            file_path="README.md",
            start_line=1,
            end_line=30,
            content="# HTTPX: A next-generation HTTP client for Python",
            source_type=DocumentType.DOCUMENTATION,
        ),
    ]


def test_extract_citations_various_formats():
    text = (
        "The URL class is declared in [httpx/_urls.py L20-L40]. "
        "Also see (_client.py, lines 110-125) for dispatching. "
        "For features, check `README.md`, L5-L25, or click [README](README.md#L1-L15)."
    )
    citations = CitationVerifier.extract_citations(text)
    assert len(citations) == 4

    # 1. Bracket format
    assert citations[0][1] == "httpx/_urls.py"
    assert citations[0][2] == 20
    assert citations[0][3] == 40

    # 2. Paren format
    assert citations[1][1] == "_client.py"
    assert citations[1][2] == 110
    assert citations[1][3] == 125

    # 3. Backtick format
    assert citations[2][1] == "README.md"
    assert citations[2][2] == 5
    assert citations[2][3] == 25

    # 4. Markdown link format
    assert citations[3][1] == "README.md"
    assert citations[3][2] == 1
    assert citations[3][3] == 15


def test_verify_exact_matches(sample_chunks):
    text = "The URL class is declared in [httpx/_urls.py L15-L35]."
    report = CitationVerifier.verify_and_patch(text, sample_chunks)

    assert report.is_fully_verified is True
    assert report.total_citations == 1
    assert report.verified_count == 1
    assert report.drifted_count == 0
    assert report.hallucinated_count == 0
    assert report.citations[0].status == CitationStatus.VERIFIED
    assert report.corrected_text == text


def test_drifted_line_coordinates_auto_repaired(sample_chunks):
    # LLM cited L200-L240, but the chunk on disk is actually L10-L50
    text = "The URL class is parsed in [httpx/_urls.py L200-L240]."
    report = CitationVerifier.verify_and_patch(text, sample_chunks)

    assert report.is_fully_verified is False
    assert report.total_citations == 1
    assert report.verified_count == 0
    assert report.drifted_count == 1
    assert report.citations[0].status == CitationStatus.DRIFTED
    assert report.citations[0].actual_start_line == 10
    assert report.citations[0].actual_end_line == 50

    # Verify that the text was patched automatically with chunk boundaries
    assert "[httpx-0.27.0/httpx/_urls.py L10-L50]" in report.corrected_text
    assert "L200-L240" not in report.corrected_text


def test_hallucinated_file_detected(sample_chunks):
    text = "Redis caching is implemented in [redis_backend.py L10-L30]."
    report = CitationVerifier.verify_and_patch(text, sample_chunks)

    assert report.is_fully_verified is False
    assert report.total_citations == 1
    assert report.hallucinated_count == 1
    assert report.citations[0].status == CitationStatus.HALLUCINATED
    assert "was not found in any retrieved context chunk" in report.citations[0].reason


def test_subpath_and_basename_matching(sample_chunks):
    # Suffix match: "_urls.py" matches "httpx-0.27.0/httpx/_urls.py"
    text = "Check [_urls.py L15-L25] for url parsing."
    report = CitationVerifier.verify_and_patch(text, sample_chunks)

    assert report.verified_count == 1
    assert report.citations[0].status == CitationStatus.VERIFIED
    assert report.citations[0].matched_chunk_id == "chunk_urls_1"
