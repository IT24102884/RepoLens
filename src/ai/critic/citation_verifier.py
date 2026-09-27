from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from ai.core.models import DocumentChunk


class CitationStatus(str, Enum):
    """Status of an extracted citation coordinate."""
    VERIFIED = "VERIFIED"          # File and line coordinates match retrieved chunks
    DRIFTED = "DRIFTED"            # File exists in retrieved chunks, but line numbers drifted outside chunk
    HALLUCINATED = "HALLUCINATED"  # File was not found in any retrieved chunks


class VerifiedCitation(BaseModel):
    """Detailed verification record for a single citation instance."""
    raw_citation: str
    file_path: str
    cited_start_line: int
    cited_end_line: int
    status: CitationStatus
    actual_start_line: Optional[int] = None
    actual_end_line: Optional[int] = None
    matched_chunk_id: Optional[str] = None
    corrected_citation: Optional[str] = None
    reason: str


class CitationVerificationReport(BaseModel):
    """Aggregate report of all citations in a generated draft."""
    is_fully_verified: bool
    total_citations: int
    verified_count: int
    drifted_count: int
    hallucinated_count: int
    citations: List[VerifiedCitation] = Field(default_factory=list)
    corrected_text: str


class CitationVerifier:
    """Deterministic Citation & Line Coordinate Verifier (System D - Component 1).
    
    Validates every file and line coordinate cited in a generated response against the actual
    retrieved DocumentChunks without invoking an LLM (0ms latency, $0.00 cost).
    Automatically repairs drifted line coordinates by snapping them to verbatim chunk boundaries.
    """

    # Matches [Source 1: path/file.py (L10-L25)], [file.py L10-L20], [file.py:10-20], [file.py (L10-L20)]
    PATTERN_BRACKET = re.compile(
        r"\[(?:Source\s+\d+:\s*)?([A-Za-z0-9_\-\./\\]+\.[a-zA-Z0-9]+)\s*(?:\(|:|,|\s)?\s*[Ll]?(\d+)(?:[–\-][Ll]?(\d+))?\s*\)?\]"
    )

    # Matches (path/file.py, line 10-20), (path/file.py:10-20), path/file.py, lines 10–20
    PATTERN_PAREN = re.compile(
        r"\(([A-Za-z0-9_\-\./\\]+\.[a-zA-Z0-9]+)\s*(?:,?\s*lines?\s*|:)\s*(\d+)(?:[–\-][Ll]?(\d+))?\)"
    )

    # Matches `path/file.py`, L10-L20 or `path/file.py` (L10-L20)
    PATTERN_BACKTICK = re.compile(
        r"`([A-Za-z0-9_\-\./\\]+\.[a-zA-Z0-9]+)`\s*(?:,|\s)?\s*\(?[Ll]?(\d+)(?:[–\-][Ll]?(\d+))?\)?"
    )

    # Matches markdown link with line anchor: [label](path/file.py#L10-L20) or [label](file:///path/file.py#L10-L20)
    PATTERN_MD_LINK = re.compile(
        r"\[([^\]]+)\]\((?:file:\/\/\/)?([A-Za-z0-9_\-\./\\]+\.[a-zA-Z0-9]+)#L(\d+)(?:-L?(\d+))?\)"
    )

    @classmethod
    def _normalize_path(cls, path_str: str) -> str:
        """Normalize directory separators and trim whitespace."""
        return path_str.replace("\\", "/").strip().lstrip("./")

    @classmethod
    def _match_chunk(cls, cited_file: str, chunks: List[DocumentChunk]) -> Optional[DocumentChunk]:
        """Find the best matching DocumentChunk for a cited file path."""
        norm_cited = cls._normalize_path(cited_file)
        cited_name = Path(norm_cited).name.lower()

        # 1. Exact match on normalized path
        for chunk in chunks:
            norm_chunk = cls._normalize_path(chunk.file_path)
            if norm_cited.lower() == norm_chunk.lower():
                return chunk

        # 2. Suffix / subpath match (e.g. "httpx/_urls.py" in "httpx-0.27.0/httpx/_urls.py")
        for chunk in chunks:
            norm_chunk = cls._normalize_path(chunk.file_path)
            if norm_chunk.lower().endswith(norm_cited.lower()) or norm_cited.lower().endswith(norm_chunk.lower()):
                return chunk

        # 3. Basename match (e.g. "_urls.py" == "_urls.py" or "README.md" == "README.md")
        for chunk in chunks:
            chunk_name = Path(chunk.file_path).name.lower()
            if cited_name == chunk_name:
                return chunk

        return None

    @classmethod
    def extract_citations(cls, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract all (raw_match, file_path, start_line, end_line) citations from text."""
        citations: List[Tuple[str, str, int, int]] = []
        seen_spans = set()

        def add_match(match_span, raw, path, start_s, end_s):
            if match_span in seen_spans:
                return
            seen_spans.add(match_span)
            start_num = int(start_s)
            end_num = int(end_s) if end_s else start_num
            citations.append((raw, path, min(start_num, end_num), max(start_num, end_num)))

        # 1. Check bracket citations [file.py L10-L20]
        for m in cls.PATTERN_BRACKET.finditer(text):
            add_match(m.span(), m.group(0), m.group(1), m.group(2), m.group(3))

        # 2. Check paren citations (file.py, line 10-20)
        for m in cls.PATTERN_PAREN.finditer(text):
            add_match(m.span(), m.group(0), m.group(1), m.group(2), m.group(3))

        # 3. Check backtick citations `file.py`, L10-L20
        for m in cls.PATTERN_BACKTICK.finditer(text):
            add_match(m.span(), m.group(0), m.group(1), m.group(2), m.group(3))

        # 4. Check markdown links [label](file.py#L10-L20)
        for m in cls.PATTERN_MD_LINK.finditer(text):
            add_match(m.span(), m.group(0), m.group(2), m.group(3), m.group(4))

        return citations

    @classmethod
    def verify_and_patch(
        cls, draft_text: str, retrieved_chunks: List[DocumentChunk]
    ) -> CitationVerificationReport:
        """Verify all citations in draft text against retrieved chunks and snap drifted coordinates."""
        raw_citations = cls.extract_citations(draft_text)

        verified_records: List[VerifiedCitation] = []
        verified_count = 0
        drifted_count = 0
        hallucinated_count = 0

        # Map raw text substrings to their corrected strings for patching
        patches: Dict[str, str] = {}

        for raw_citation, file_path, cited_start, cited_end in raw_citations:
            matched_chunk = cls._match_chunk(file_path, retrieved_chunks)

            if matched_chunk is None:
                # File not found in any retrieved chunk
                hallucinated_count += 1
                verified_records.append(
                    VerifiedCitation(
                        raw_citation=raw_citation,
                        file_path=file_path,
                        cited_start_line=cited_start,
                        cited_end_line=cited_end,
                        status=CitationStatus.HALLUCINATED,
                        reason=f"File '{file_path}' was not found in any retrieved context chunk.",
                    )
                )
                continue

            # File was found in retrieved chunks; check line coordinate overlap
            chunk_start = matched_chunk.start_line
            chunk_end = matched_chunk.end_line

            # Check if cited line range overlaps with chunk line boundaries
            is_overlapping = not (cited_end < chunk_start or cited_start > chunk_end)

            if is_overlapping:
                # Fully verified line coordinates
                verified_count += 1
                verified_records.append(
                    VerifiedCitation(
                        raw_citation=raw_citation,
                        file_path=file_path,
                        cited_start_line=cited_start,
                        cited_end_line=cited_end,
                        status=CitationStatus.VERIFIED,
                        actual_start_line=chunk_start,
                        actual_end_line=chunk_end,
                        matched_chunk_id=matched_chunk.id,
                        reason="Coordinates align with retrieved chunk bounds.",
                    )
                )
            else:
                # Line drifted outside chunk boundaries -> Auto-repair coordinate
                drifted_count += 1
                corrected = f"[{matched_chunk.file_path} L{chunk_start}-L{chunk_end}]"
                patches[raw_citation] = corrected
                verified_records.append(
                    VerifiedCitation(
                        raw_citation=raw_citation,
                        file_path=file_path,
                        cited_start_line=cited_start,
                        cited_end_line=cited_end,
                        status=CitationStatus.DRIFTED,
                        actual_start_line=chunk_start,
                        actual_end_line=chunk_end,
                        matched_chunk_id=matched_chunk.id,
                        corrected_citation=corrected,
                        reason=f"Cited lines ({cited_start}-{cited_end}) drifted outside chunk ({chunk_start}-{chunk_end}). Snapped to chunk bounds.",
                    )
                )

        # Apply deterministic text patches for drifted coordinates
        patched_text = draft_text
        for old_str, new_str in patches.items():
            patched_text = patched_text.replace(old_str, new_str)

        is_fully_verified = (hallucinated_count == 0 and drifted_count == 0)

        return CitationVerificationReport(
            is_fully_verified=is_fully_verified,
            total_citations=len(raw_citations),
            verified_count=verified_count,
            drifted_count=drifted_count,
            hallucinated_count=hallucinated_count,
            citations=verified_records,
            corrected_text=patched_text,
        )
