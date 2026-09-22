import pytest
from pydantic import ValidationError
from src.core.models import DocumentChunk, DocumentType


def test_document_chunk_deterministic_id():
    """Verify identical content produces the exact same ID."""
    chunk1 = DocumentChunk.create(
        source_type=DocumentType.CODE,
        file_path="httpx/_client.py",
        start_line=10,
        end_line=20,
        content="def send_request(): pass",
    )

    chunk2 = DocumentChunk.create(
        source_type=DocumentType.CODE,
        file_path="httpx/_client.py",
        start_line=10,
        end_line=20,
        content="def send_request(): pass",
    )

    # Both IDs MUST be identical
    assert chunk1.id == chunk2.id
    assert len(chunk1.id) == 16


def test_document_chunk_hash_changes_on_content_diff():
    """Verify changing content creates a distinct ID."""
    base_chunk = DocumentChunk.create(
        source_type=DocumentType.CODE,
        file_path="httpx/_client.py",
        start_line=10,
        end_line=20,
        content="def send_request(): pass",
    )

    modified_chunk = DocumentChunk.create(
        source_type=DocumentType.CODE,
        file_path="httpx/_client.py",
        start_line=10,
        end_line=20,
        content="def send_request(): return 200",  # Changed!
    )

    assert base_chunk.id != modified_chunk.id


def test_document_chunk_rejects_negative_line_numbers():
    """Verify Pydantic validation rejects impossible line numbers."""
    with pytest.raises(ValidationError):
        DocumentChunk(
            id="test-id",
            source_type=DocumentType.CODE,
            file_path="test.py",
            start_line=-5,  # we are testing by putting a invalid negative line number
            end_line=10,
            content="print('hello')",
        )