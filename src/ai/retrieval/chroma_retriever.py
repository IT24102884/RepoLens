import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from ai.core.models import DocumentChunk, DocumentType

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


class ChromaRetriever:
    """High-throughput local Vector Retriever backed by ChromaDB and ONNX embeddings."""

    def __init__(
        self,
        persist_directory: Path = Path("data/chroma_storage"),
        collection_name: str = "httpx_knowledge",
        chunks_path: Path = Path("data/processed/chunks.jsonl"),
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.chunks_path = Path(chunks_path)

        # 1. Initialize persistent ChromaDB storage
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))

        # 2. Use Chroma's native local embedding engine (all-MiniLM-L6-v2 via ONNX)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.chunks: List[DocumentChunk] = []
        self._load_chunks()

    def _load_chunks(self):
        """Read DocumentChunks from JSONL."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}")

        self.chunks = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(DocumentChunk.model_validate_json(line))

    def index_chunks(self, batch_size: int = 200, force_reindex: bool = False) -> None:
        """Index chunks locally with zero API rate limits."""
        if force_reindex:
            print("[*] Force re-indexing: clearing existing collection...")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        existing_count = self.collection.count()
        if existing_count >= len(self.chunks) and not force_reindex:
            print(f"[*] ChromaDB collection '{self.collection_name}' already contains {existing_count} chunks. Ready to search.")
            return

        # Checkpoint: find chunks not yet in ChromaDB
        existing_data = self.collection.get()
        existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()
        chunks_to_index = [c for c in self.chunks if c.id not in existing_ids]

        print(f"[*] Indexing {len(chunks_to_index)} chunks locally via ChromaDB ONNX embeddings...")

        for i in range(0, len(chunks_to_index), batch_size):
            batch = chunks_to_index[i : i + batch_size]
            ids = [c.id for c in batch]
            # Include file path + symbol context in the document text for high semantic accuracy
            documents = [f"File: {c.file_path}\n{c.content}" for c in batch]
            metadatas = [
                {
                    "source_type": c.source_type.value,
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "symbol_name": str(c.metadata.get("symbol_name", "")),
                    "parent_class": str(c.metadata.get("parent_class", "")),
                    "header_path": str(c.metadata.get("header_path", "")),
                }
                for c in batch
            ]

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            current_total = len(existing_ids) + min(i + batch_size, len(chunks_to_index))
            print(f"    Indexed {current_total}/{len(self.chunks)} chunks...")

        print(f"[+] All {self.collection.count()} chunks indexed successfully in ChromaDB at {self.persist_directory}!\n")

    @traceable(name="Chroma Dense Search", run_type="retriever")
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[DocumentType] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Query ChromaDB with automatic local query embedding."""
        where_filter = None
        if filter_type:
            where_filter = {"source_type": filter_type.value}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"],
        )

        output: List[Tuple[DocumentChunk, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        documents = results["documents"][0] if results["documents"] else [""] * len(ids)

        for chunk_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            similarity = max(0.0, 1.0 - dist)
            # Strip the added "File: ...\n" prefix for clean content
            clean_content = doc.split("\n", 1)[1] if doc.startswith("File: ") and "\n" in doc else doc

            chunk = DocumentChunk(
                id=chunk_id,
                source_type=DocumentType(meta.get("source_type", "CODE")),
                file_path=meta.get("file_path", "unknown"),
                start_line=meta.get("start_line", 1),
                end_line=meta.get("end_line", 1),
                content=clean_content,
                metadata=meta,
            )
            output.append((chunk, similarity))

        return output