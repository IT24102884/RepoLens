import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from src.core.models import DocumentChunk, DocumentType
from src.ingestion.code_chunker import ASTCodeChunker
from src.ingestion.doc_chunker import MarkdownDocChunker
from src.ingestion.ticket_chunker import TicketChunker

# Ignored directory names during scanning
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "node_modules"}
# Ignored file extensions
IGNORED_EXTENSIONS = {".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".lock"}


class IngestionPipeline:
    """Orchestrates multi-source scanning, routing, chunking, and persistence."""

    def __init__(self):
        self.code_chunker = ASTCodeChunker()
        self.doc_chunker = MarkdownDocChunker()
        self.ticket_chunker = TicketChunker()

    def process_file(self, file_path: Path, repo_root: Path) -> List[DocumentChunk]:
        """Route a single file to its appropriate chunker based on extension."""
        relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        # Route by extension
        ext = file_path.suffix.lower()
        if ext == ".py":
            return self.code_chunker.chunk(content, relative_path)
        elif ext in {".md", ".markdown", ".rst"}:
            return self.doc_chunker.chunk(content, relative_path)
        elif ext == ".json" and "issue" in relative_path.lower():
            return self.ticket_chunker.chunk(content, relative_path)

        return []

    def process_directory(self, target_dir: Path) -> List[DocumentChunk]:
        """Recursively walk a directory and extract chunks from all supported sources."""
        all_chunks: List[DocumentChunk] = []
        target_dir = Path(target_dir).resolve()

        for root, dirs, files in os.walk(target_dir):
            # Skip ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.suffix.lower() in IGNORED_EXTENSIONS:
                    continue

                chunks = self.process_file(file_path, repo_root=target_dir)
                all_chunks.extend(chunks)

        return all_chunks

    def save_chunks(self, chunks: List[DocumentChunk], output_path: Path) -> None:
        """Write all chunks into a newline-delimited JSON (JSONL) file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

    def get_summary(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Compute summary statistics across all ingested chunks."""
        summary = {
            "total_chunks": len(chunks),
            "code_chunks": sum(1 for c in chunks if c.source_type == DocumentType.CODE),
            "doc_chunks": sum(1 for c in chunks if c.source_type == DocumentType.DOCUMENTATION),
            "ticket_chunks": sum(1 for c in chunks if c.source_type == DocumentType.ISSUE_PR),
        }
        return summary