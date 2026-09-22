from src.core.models import DocumentType
from src.ingestion.doc_chunker import MarkdownDocChunker


def test_markdown_headers():
    # 1. Test header hierarchy
    text = (
        "# HTTPX\n\n"
        "An HTTP library.\n\n"
        "## Quickstart\n\n"
        "Usage guide here.\n"
    )
    chunker = MarkdownDocChunker()
    chunks = chunker.chunk(content=text, file_path="docs/index.md")

    
    assert len(chunks) == 2
    assert chunks[0].metadata["header_path"] == "HTTPX"
    assert chunks[1].metadata["header_path"] == "HTTPX > Quickstart"
    assert chunks[0].source_type == DocumentType.DOCUMENTATION


def test_markdown_code_fence_not_split():
    # 2. Test code blocks are kept intact
    code_block = "```python\n# This is a comment\nx = 1\n```"
    text = f"# Title\n\n{code_block}\n"
    
    chunker = MarkdownDocChunker()
    chunks = chunker.chunk(content=text, file_path="docs/test.md")

    assert len(chunks) == 1
    assert "# This is a comment" in chunks[0].content