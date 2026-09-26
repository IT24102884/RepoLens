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


def test_treesitter_typescript_chunking():
    ts_code = """interface User {
    id: number;
    name: string;
}

export function getUser(id: number): User {
    return { id: id, name: "Alice" };
}

class UserService {
    private users: User[] = [];
}
"""
    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=ts_code, file_path="services/user.ts")

    assert len(chunks) == 3
    names = [c.metadata.get("symbol_name") for c in chunks]
    assert "User" in names
    assert "getUser" in names
    assert "UserService" in names
    assert chunks[0].metadata["language"] == "typescript"
    assert chunks[1].start_line == 6


def test_treesitter_go_chunking():
    go_code = """package main

type Config struct {
    Port int
}

func StartServer(cfg Config) error {
    return nil
}
"""
    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=go_code, file_path="server.go")

    assert len(chunks) == 2
    names = [c.metadata.get("symbol_name") for c in chunks]
    assert "Config" in names
    assert "StartServer" in names
    assert chunks[0].metadata["language"] == "go"


def test_treesitter_java_chunking():
    java_code = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=java_code, file_path="Calculator.java")

    assert len(chunks) >= 1
    assert chunks[0].metadata["language"] == "java"
    assert chunks[0].metadata["symbol_name"] == "Calculator"


def test_ast_code_chunker_windowed_fallback_large_file():
    """Verify that a 130-line file with syntax errors is chunked into 60-line windows with 10-line overlap."""
    lines = [f"broken_code_line_{i} = {i} +" for i in range(1, 131)]
    broken_content = "\n".join(lines)

    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=broken_content, file_path="broken_large.py")

    assert len(chunks) == 3
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 60
    assert chunks[1].start_line == 51
    assert chunks[1].end_line == 110
    assert chunks[2].start_line == 101
    assert chunks[2].end_line == 130
    assert chunks[0].metadata["chunk_type"] == "sliding_window"
    assert chunks[0].metadata["fallback"] is True


def test_ast_code_chunker_unmapped_language_fallback():
    """Files in languages without Tree-sitter parsers (e.g. Kotlin .kt) cleanly fallback to windowed chunks."""
    kt_code = "\n".join([f"val item{i}: String = \"value_{i}\"" for i in range(1, 75)])
    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=kt_code, file_path="app/Main.kt")

    assert len(chunks) == 2
    assert chunks[0].metadata["language"] == "kt"
    assert chunks[0].metadata["chunk_type"] == "sliding_window"
    assert chunks[1].start_line == 51
    assert chunks[1].end_line == 74