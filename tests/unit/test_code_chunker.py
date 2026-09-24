from ai.core.models import DocumentType
from ai.ingestion.code_chunker import ASTCodeChunker

SAMPLE_PYTHON_CODE = '''"""This is a sample module for testing."""

class HTTPClient:
    """A client for sending HTTP requests."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    @classmethod
    def create_default(cls):
        """Factory method with decorator."""
        return cls(base_url="https://api.example.com")


async def fetch_status(client: HTTPClient) -> int:
    """Standalone async function."""
    return 200
'''


def test_ast_code_chunker_parses_classes_and_methods():
    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=SAMPLE_PYTHON_CODE, file_path="sample.py")

    assert len(chunks) == 5

    # 2. Check Module Docstring
    doc_chunk = chunks[0]
    assert doc_chunk.metadata["symbol_type"] == "module_docstring"
    assert "sample module" in doc_chunk.content

    # 3. Check Class Header
    class_chunk = chunks[1]
    assert class_chunk.metadata["symbol_name"] == "HTTPClient"
    assert class_chunk.metadata["symbol_type"] == "class_header"

    # 4. Check Method & Parent Class Metadata
    init_chunk = chunks[2]
    assert init_chunk.metadata["symbol_name"] == "__init__"
    assert init_chunk.metadata["parent_class"] == "HTTPClient"

    # 5. Check Decorator Preservation
    decorated_chunk = chunks[3]
    assert decorated_chunk.metadata["symbol_name"] == "create_default"
    assert "@classmethod" in decorated_chunk.content  # Decorator was NOT chopped off!
    assert decorated_chunk.metadata["parent_class"] == "HTTPClient"

    # 6. Check Standalone Async Function
    async_chunk = chunks[4]
    assert async_chunk.metadata["symbol_name"] == "fetch_status"
    assert async_chunk.metadata["is_async"] is True
    assert "parent_class" not in async_chunk.metadata  # Standalone function


def test_ast_code_chunker_handles_syntax_error():
    """If a file contains broken Python syntax, it should gracefully fallback to 1 chunk."""
    broken_code = "def broken_func(:\n    pass"
    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=broken_code, file_path="broken.py")

    assert len(chunks) == 1
    assert chunks[0].metadata.get("fallback") is True