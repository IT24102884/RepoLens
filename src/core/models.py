from enum import Enum
import hashlib
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported source types in the engineering knowledge base."""
    CODE = "CODE"
    DOCUMENTATION = "DOCUMENTATION"
    ISSUE_PR = "ISSUE_PR"


class DocumentChunk(BaseModel):
    """Normalized unit of retrieval and citation across all sources."""
    id: str = Field(description="Unique deterministic chunk ID")
    source_type: DocumentType = Field(description="Type of knowledge source")
    file_path: str = Field(description="Relative path of source file or issue key")
    start_line: int = Field(ge=1, description="1-indexed starting line")
    end_line: int = Field(ge=1, description="1-indexed ending line")
    content: str = Field(description="Clean text or code chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific metadata")
    token_count: Optional[int] = Field(default=None, ge=0, description="Estimated or exact token count")

    @classmethod
    def create(
        cls,
        source_type: DocumentType,
        file_path: str,
        start_line: int,
        end_line: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
    ) -> "DocumentChunk":
        """Factory method that automatically generates a deterministic SHA-256 ID."""
        normalized_path = file_path.replace("\\", "/").strip("/")
        raw_key = f"{normalized_path}:{start_line}:{end_line}:{content.strip()}"
        chunk_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

        return cls(
            id=chunk_id,
            source_type=source_type,
            file_path=normalized_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            metadata=metadata or {},
            token_count=token_count,
        )