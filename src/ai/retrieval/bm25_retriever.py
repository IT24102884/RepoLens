import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi

from ai.core.models import DocumentChunk, DocumentType

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


def tokenize_code(text: str) -> List[str]:
    """Code-aware tokenizer preserving programming syntax and identifiers."""
    tokens: List[str] = []
    
    # 1. Match issue numbers like #1240 or #1405
    issue_tags = re.findall(r'#\d+', text)
    for tag in issue_tags:
        tokens.append(tag.lower())
        tokens.append(tag[1:])
    
    # 2. Extract words, identifiers, and dotted paths
    raw_identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b', text)
    for ident in raw_identifiers:
        ident_lower = ident.lower()
        tokens.append(ident_lower)
        
        parts = ident.split(".") if "." in ident else [ident]
        for part in parts:
            part_lower = part.lower()
            if part_lower != ident_lower:
                tokens.append(part_lower)
            
            if "_" in part:
                for sub in part.split("_"):
                    if sub:
                        tokens.append(sub.lower())
                        
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', part)
            if len(camel_parts) > 1:
                for cp in camel_parts:
                    tokens.append(cp.lower())
                
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    tokens.extend(words)
    
    seen = set()
    deduped = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            deduped.append(tok)
            
    return deduped


class BM25Retriever:
    """Exact-match sparse keyword retriever using BM25Okapi over code and tickets."""

    def __init__(self, chunks_path: Path = Path("data/processed/chunks.jsonl")):
        self.chunks_path = Path(chunks_path)
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        """Load chunks and build BM25Okapi inverted index."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}")

        self.chunks = []
        self.tokenized_corpus = []

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = DocumentChunk.model_validate_json(line)
                    self.chunks.append(chunk)
                    
                    indexing_text = (
                        f"{chunk.file_path}\n"
                        f"{json.dumps(chunk.metadata)}\n"
                        f"{chunk.content}"
                    )
                    tokens = tokenize_code(indexing_text)
                    self.tokenized_corpus.append(tokens)

        # b=0.3 is optimal for code chunking to prevent penalizing full class blocks
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.5, b=0.3)

    @traceable(name="BM25 Sparse Search", run_type="retriever")
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[DocumentType] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Query the BM25 index with code-aware tokenization."""
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
                
            chunk = self.chunks[idx]
            if filter_type and chunk.source_type != filter_type:
                continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results
