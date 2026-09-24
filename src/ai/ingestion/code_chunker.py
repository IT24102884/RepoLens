import ast
from typing import Any, Dict, List, Optional
from ai.core.models import DocumentChunk, DocumentType
from ai.ingestion.base import BaseChunker


class ASTCodeChunker(BaseChunker):
    """AST-aware chunker for Python code files."""

    def __init__(self, min_chunk_lines: int = 3):
        self.min_chunk_lines = min_chunk_lines

    def chunk(self, content: str, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Parse Python code using AST and extract classes and functions as atomic chunks."""
        chunks: List[DocumentChunk] = []
        lines = content.splitlines()

        if not content.strip():
            return chunks

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            # Fallback if file has syntax errors: emit entire file as 1 chunk
            return [
                DocumentChunk.create(
                    source_type=DocumentType.CODE,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    content=content,
                    metadata={"fallback": True},
                )
            ]

        # 1. Capture Module-Level Docstring if present
        module_docstring = ast.get_docstring(tree)
        if module_docstring and tree.body:
            first_stmt = tree.body[0]
            if hasattr(first_stmt, "end_lineno"):
                chunks.append(
                    DocumentChunk.create(
                        source_type=DocumentType.CODE,
                        file_path=file_path,
                        start_line=1,
                        end_line=first_stmt.end_lineno,
                        content="\n".join(lines[: first_stmt.end_lineno]),
                        metadata={"symbol_type": "module_docstring"},
                    )
                )

        # 2. Extract Top-Level Nodes (Classes and Standalone Functions)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._process_class(node, lines, file_path))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._process_function(node, lines, file_path, parent_class=None))

        return chunks

    def _process_class(
        self, class_node: ast.ClassDef, lines: List[str], file_path: str
    ) -> List[DocumentChunk]:
        """Extract a class summary and each of its individual methods."""
        chunks: List[DocumentChunk] = []
        class_name = class_node.name

        # Extract class signature and docstring as a summary chunk
        class_start = class_node.lineno
        doc = ast.get_docstring(class_node)

        # Class header chunk
        header_end = class_node.body[0].lineno - 1 if class_node.body else class_node.end_lineno
        class_header_lines = lines[class_start - 1 : max(class_start, header_end)]

        chunks.append(
            DocumentChunk.create(
                source_type=DocumentType.CODE,
                file_path=file_path,
                start_line=class_start,
                end_line=max(class_start, header_end),
                content="\n".join(class_header_lines),
                metadata={
                    "symbol_name": class_name,
                    "symbol_type": "class_header",
                    "docstring": doc,
                },
            )
        )

        # Extract each method in the class as its own independent chunk
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(
                    self._process_function(item, lines, file_path, parent_class=class_name)
                )

        return chunks

    def _process_function(
        self,
        func_node: Any,
        lines: List[str],
        file_path: str,
        parent_class: Optional[str] = None,
    ) -> DocumentChunk:
        """Extract a function or method preserving its complete lines and decorators."""
        start_line = func_node.lineno
        end_line = func_node.end_lineno or start_line

        # Include decorators if present
        if func_node.decorator_list:
            start_line = min(d.lineno for d in func_node.decorator_list)

        func_content = "\n".join(lines[start_line - 1 : end_line])
        docstring = ast.get_docstring(func_node)

        metadata: Dict[str, Any] = {
            "symbol_name": func_node.name,
            "symbol_type": "method" if parent_class else "function",
            "is_async": isinstance(func_node, ast.AsyncFunctionDef),
            "docstring": docstring,
        }
        if parent_class:
            metadata["parent_class"] = parent_class

        return DocumentChunk.create(
            source_type=DocumentType.CODE,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=func_content,
            metadata=metadata,
        )