from src.ingestion.code_chunker import ASTCodeChunker

# 1. Sample Python Code to Chunk
SAMPLE_CODE = '''"""Client module for HTTP requests."""

class APIClient:
    """Main client class."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    @classmethod
    def default_client(cls):
        """Creates default client."""
        return cls("https://api.example.com")


def check_health() -> bool:
    """Standalone health check function."""
    return True
'''

def main():
    print("\n" + "=" * 60)
    print("       AST PYTHON CODE CHUNKING DEMO")
    print("=" * 60)

    chunker = ASTCodeChunker()
    chunks = chunker.chunk(content=SAMPLE_CODE, file_path="client.py")

    print(f"\nTotal Chunks Created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"--- [CHUNK {i}] " + "-" * 40)
        print(f"ID:          {chunk.id}")
        print(f"Type:        {chunk.source_type.value}")
        print(f"File & Lines:{chunk.file_path} (Lines {chunk.start_line} -> {chunk.end_line})")
        print(f"Metadata:    {chunk.metadata}")
        print("-" * 55)
        print("Content:")
        for line in chunk.content.splitlines():
            print(f"  {line}")
        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()