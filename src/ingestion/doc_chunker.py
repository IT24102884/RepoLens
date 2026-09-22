import re
from typing import Any, Dict, List, Optional
from src.core.models import DocumentChunk, DocumentType
from src.ingestion.base import BaseChunker


class MarkdownDocChunker(BaseChunker):
    """Header-aware and code-block protective chunker for Markdown documents."""

    def __init__(self, target_chunk_lines: int = 40):
        self.target_chunk_lines = target_chunk_lines
        self.header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    def chunk(self, content: str, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Split markdown content by headers while preserving code fences and breadcrumbs."""
        chunks: List[DocumentChunk] = []
        lines = content.splitlines()

        if not content.strip():
            return chunks

        current_headers: Dict[int, str] = {}  # level -> title
        current_chunk_lines: List[str] = []
        chunk_start_line = 1
        in_code_block = False

        for line_idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 1. Track Code Fences (Never split inside code blocks!)
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                current_chunk_lines.append(line)
                continue

            # 2. Detect Markdown Headers (only when NOT inside a code fence)
            header_match = self.header_pattern.match(line) if not in_code_block else None

            if header_match:
                header_level = len(header_match.group(1))
                header_title = header_match.group(2).strip()

                # If we have accumulated text, flush it as a chunk before starting a new section
                if current_chunk_lines and any(l.strip() for l in current_chunk_lines):
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        chunks.append(
                            self._create_doc_chunk(
                                lines=current_chunk_lines,
                                file_path=file_path,
                                start_line=chunk_start_line,
                                end_line=line_idx - 1,
                                headers=current_headers,
                            )
                        )
                    current_chunk_lines = []
                    chunk_start_line = line_idx

                # Update the header breadcrumb hierarchy
                # Remove any deeper headers (e.g. if we see an H2, clear previous H3/H4)
                current_headers = {lvl: title for lvl, title in current_headers.items() if lvl < header_level}
                current_headers[header_level] = header_title

            current_chunk_lines.append(line)

        # Flush any remaining lines at the end of the file
        if current_chunk_lines and any(l.strip() for l in current_chunk_lines):
            chunks.append(
                self._create_doc_chunk(
                    lines=current_chunk_lines,
                    file_path=file_path,
                    start_line=chunk_start_line,
                    end_line=len(lines),
                    headers=current_headers,
                )
            )

        return chunks

    def _create_doc_chunk(
        self,
        lines: List[str],
        file_path: str,
        start_line: int,
        end_line: int,
        headers: Dict[int, str],
    ) -> DocumentChunk:
        """Create a DocumentChunk with header hierarchy attached to metadata."""
        # Build breadcrumb trail: "Client Configuration > Timeouts > Connect"
        sorted_levels = sorted(headers.keys())
        breadcrumb = " > ".join(headers[lvl] for lvl in sorted_levels) if sorted_levels else "Overview"
        top_header = headers[sorted_levels[-1]] if sorted_levels else "Overview"

        metadata: Dict[str, Any] = {
            "header_path": breadcrumb,
            "section_title": top_header,
        }

        return DocumentChunk.create(
            source_type=DocumentType.DOCUMENTATION,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content="\n".join(lines).strip(),
            metadata=metadata,
        )