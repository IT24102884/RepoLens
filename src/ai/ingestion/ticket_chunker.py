import json
from typing import Any, Dict, List
from ai.core.models import DocumentChunk, DocumentType
from ai.ingestion.base import BaseChunker


class TicketChunker(BaseChunker):
    """Chunker for GitHub issues, PRs, and engineering support tickets."""

    def chunk(self, content: str, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Parse structured ticket JSON into an issue summary and thread chunks."""
        chunks: List[DocumentChunk] = []

        try:
            ticket_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback if content is plain text rather than structured JSON
            return [
                DocumentChunk.create(
                    source_type=DocumentType.ISSUE_PR,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(content.splitlines()),
                    content=content,
                    metadata={"fallback": True},
                )
            ]

        issue_id = str(ticket_data.get("number", "unknown"))
        title = ticket_data.get("title", "Untitled")
        author = ticket_data.get("author", "unknown")
        state = ticket_data.get("state", "open")
        labels = ticket_data.get("labels", [])
        body = ticket_data.get("body", "").strip()
        comments = ticket_data.get("comments", [])

        # 1. Chunk 1: The Primary Issue Summary
        summary_content = (
            f"Issue #{issue_id}: {title}\n"
            f"Author: @{author} | State: {state.upper()} | Labels: {', '.join(labels)}\n\n"
            f"Description:\n{body}"
        )

        chunks.append(
            DocumentChunk.create(
                source_type=DocumentType.ISSUE_PR,
                file_path=f"issues/{issue_id}",
                start_line=1,
                end_line=len(summary_content.splitlines()),
                content=summary_content,
                metadata={
                    "issue_id": issue_id,
                    "title": title,
                    "author": author,
                    "state": state,
                    "labels": labels,
                    "is_header": True,
                },
            )
        )

        # 2. Chunk 2+: Group Comments into Discussion Chunks
        if comments:
            comment_lines: List[str] = []
            start_comment_idx = 1

            for idx, comment in enumerate(comments, start=1):
                c_author = comment.get("author", "user")
                c_body = comment.get("body", "").strip()
                comment_lines.append(f"Comment from @{c_author}:\n{c_body}\n")

                # Emit a discussion chunk every 3 comments or at the end
                if len(comment_lines) >= 3 or idx == len(comments):
                    full_discussion = (
                        f"Discussion on Issue #{issue_id} ({title}):\n\n"
                        + "\n".join(comment_lines)
                    )
                    chunks.append(
                        DocumentChunk.create(
                            source_type=DocumentType.ISSUE_PR,
                            file_path=f"issues/{issue_id}#comments-{start_comment_idx}-{idx}",
                            start_line=1,
                            end_line=len(full_discussion.splitlines()),
                            content=full_discussion,
                            metadata={
                                "issue_id": issue_id,
                                "title": title,
                                "is_discussion": True,
                                "comment_range": f"{start_comment_idx}-{idx}",
                            },
                        )
                    )
                    comment_lines = []
                    start_comment_idx = idx + 1

        return chunks