from abc import ABC, abstractmethod
from typing import List
from src.core.models import DocumentChunk


class BaseChunker(ABC):
    """Abstract interface for all multi-source chunkers."""

    @abstractmethod
    def chunk(self, content: str, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Parse raw content and return normalized DocumentChunks."""
        pass