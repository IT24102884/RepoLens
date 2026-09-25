from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's technical question about the repository")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    system: str = Field(default="c", description="System architecture: 'a' for Baseline Dense RAG, 'b' for Hybrid BM25+RRF, 'c' for Routed Hybrid RAG")


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
    system_version: str = "System C (Cascading Intent Router)"
    intent: Optional[str] = None
    route_source: Optional[str] = None


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


class IngestRepoRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL, e.g. 'https://github.com/tiangolo/fastapi'")
    reindex: bool = Field(default=False, description="Whether to force re-cloning and re-indexing")


class IngestRepoResponse(BaseModel):
    status: str
    repo_name: str
    repo_slug: str
    description: str
    primary_languages: List[str]
    subsystems: List[str]
    total_files: int
    total_chunks: int
    code_chunks: int
    doc_chunks: int
    ticket_chunks: int
    message: str


class SwitchRepoRequest(BaseModel):
    repo_slug: str = Field(..., description="Repository slug to activate, e.g. 'encode_httpx' or 'default'")
