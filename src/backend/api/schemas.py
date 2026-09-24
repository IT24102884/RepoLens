from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's technical question about the repository")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class CitationItem(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    doc_type: str
    snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    system_version: str = "System A (Baseline Dense RAG)"


class RepoStatsResponse(BaseModel):
    repo_name: str
    repo_version: str
    total_chunks: int
    code_chunks: int
    doc_chunks: int
    ticket_chunks: int
    vector_store: str
    generator_model: str
    system_status: str
