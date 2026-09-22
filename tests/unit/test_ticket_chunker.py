import json
from src.core.models import DocumentType
from src.ingestion.ticket_chunker import TicketChunker


def test_ticket_chunker_parses_issue_and_comments():
    issue_payload = {
        "number": 1042,
        "title": "ConnectionTimeout on high-concurrency requests",
        "author": "alice_dev",
        "state": "closed",
        "labels": ["bug", "transport"],
        "body": "When making 100 concurrent requests, a ConnectionTimeout is raised.",
        "comments": [
            {"author": "bob_maintainer", "body": "Can you reproduce this with keepalive disabled?"},
            {"author": "alice_dev", "body": "Yes, setting keepalive_expiry=None fixed it."},
        ],
    }

    chunker = TicketChunker()
    chunks = chunker.chunk(content=json.dumps(issue_payload), file_path="issues/1042.json")

    # We expect 2 chunks: 1 Header Summary + 1 Discussion Chunk
    assert len(chunks) == 2
    assert all(c.source_type == DocumentType.ISSUE_PR for c in chunks)

    # 1. Check Header Summary Chunk
    header_chunk = chunks[0]
    assert header_chunk.metadata["issue_id"] == "1042"
    assert header_chunk.metadata["is_header"] is True
    assert "Issue #1042: ConnectionTimeout" in header_chunk.content
    assert "@alice_dev" in header_chunk.content

    # 2. Check Discussion Chunk
    disc_chunk = chunks[1]
    assert disc_chunk.metadata["issue_id"] == "1042"
    assert disc_chunk.metadata["is_discussion"] is True
    assert "bob_maintainer" in disc_chunk.content
    assert "keepalive_expiry" in disc_chunk.content


def test_ticket_chunker_handles_invalid_json():
    """Verify malformed content falls back cleanly to 1 chunk without crashing."""
    raw_text = "This is a plain bug note, not valid JSON."
    chunker = TicketChunker()
    chunks = chunker.chunk(content=raw_text, file_path="notes.txt")

    assert len(chunks) == 1
    assert chunks[0].metadata.get("fallback") is True