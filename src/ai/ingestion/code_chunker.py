import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
from ai.core.models import DocumentChunk, DocumentType
from ai.ingestion.base import BaseChunker

# Tree-sitter universal language parsers (loaded dynamically on demand)
try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False


class ASTCodeChunker(BaseChunker):
    """Universal AST-aware chunker supporting Python, TypeScript, JavaScript, Go, Rust, and Java."""

    def __init__(self, min_chunk_lines: int = 3):
        self.min_chunk_lines = min_chunk_lines

    def chunk(self, content: str, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Parse code using Tree-sitter (polyglot) or Python AST into atomic chunks."""
        chunks: List[DocumentChunk] = []
        lines = content.splitlines()

        if not content.strip():
            return chunks

        ext = Path(file_path).suffix.lower()

        # If it's a non-Python code file (TypeScript, JavaScript, Go, Rust, Java), route to Tree-sitter
        if ext != ".py":
            if HAS_TREE_SITTER:
                ts_chunks = self._chunk_with_treesitter(content, file_path, ext)
                if ts_chunks:
                    return ts_chunks
            # Fallback for unmapped grammars, missing Tree-sitter wheels, or parse failures
            return self._chunk_windowed(content, file_path, language=ext.lstrip(".") or "unknown")

        # Python files (.py) continue using our specialized Python AST parser
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            # Fallback if file has syntax errors: emit sliding window chunks
            return self._chunk_windowed(content, file_path, language="python")

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

        # 3. If no classes or functions were found (e.g. flat script, module-level execution)
        if not chunks and lines:
            return self._chunk_windowed(content, file_path, language="python")

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

        # Process class body: group non-function statements (attributes, enums, type definitions)
        current_block: List[ast.stmt] = []

        def flush_block(block: List[ast.stmt]) -> None:
            if not block:
                return
            b_start = min(st.lineno for st in block)
            b_end = max((st.end_lineno or st.lineno) for st in block)
            b_lines = lines[b_start - 1 : b_end]
            b_content = "\n".join(b_lines)
            if b_content.strip():
                chunks.append(
                    DocumentChunk.create(
                        source_type=DocumentType.CODE,
                        file_path=file_path,
                        start_line=b_start,
                        end_line=b_end,
                        content=b_content,
                        metadata={
                            "symbol_name": class_name,
                            "symbol_type": "class_body_declarations",
                            "parent_class": class_name,
                        },
                    )
                )

        # Identify if first body statement is the class docstring (already in class_header)
        first_is_docstring = False
        if class_node.body and isinstance(class_node.body[0], ast.Expr):
            val = class_node.body[0].value
            if (isinstance(val, ast.Constant) and isinstance(val.value, str)) or isinstance(val, ast.Str):
                first_is_docstring = True

        for i, item in enumerate(class_node.body):
            if i == 0 and first_is_docstring:
                continue

            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flush_block(current_block)
                current_block = []
                chunks.append(
                    self._process_function(item, lines, file_path, parent_class=class_name)
                )
            else:
                current_block.append(item)

        flush_block(current_block)
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

    def _get_parser_for_extension(self, ext: str) -> Optional[tuple]:
        """Dynamically load only the language parser needed for this file."""
        try:
            if ext == ".py":
                import tree_sitter_python as tspython
                return "python", Parser(Language(tspython.language()))

            elif ext in {".ts", ".tsx"}:
                import tree_sitter_typescript as tstypescript
                lang_fn = tstypescript.language_tsx if ext == ".tsx" else tstypescript.language_typescript
                return "typescript", Parser(Language(lang_fn()))

            elif ext in {".js", ".jsx"}:
                import tree_sitter_javascript as tsjavascript
                return "javascript", Parser(Language(tsjavascript.language()))

            elif ext == ".go":
                import tree_sitter_go as tsgo
                return "go", Parser(Language(tsgo.language()))

            elif ext == ".rs":
                import tree_sitter_rust as tsrust
                return "rust", Parser(Language(tsrust.language()))

            elif ext == ".java":
                import tree_sitter_java as tsjava
                return "java", Parser(Language(tsjava.language()))

        except Exception:
            return None
        return None

    def _chunk_with_treesitter(
        self, content: str, file_path: str, ext: str
    ) -> List[DocumentChunk]:
        """Universal AST parser for TypeScript, JavaScript, Go, Rust, Java, Python."""
        parser_info = self._get_parser_for_extension(ext)
        if not parser_info:
            return []

        lang_name, parser = parser_info
        try:
            tree = parser.parse(content.encode("utf-8"))
        except Exception:
            return []

        lines = content.splitlines()
        chunks: List[DocumentChunk] = []

        # Target declaration node types across languages
        target_types = {
            # Python
            "function_definition", "class_definition",
            # TypeScript / JavaScript
            "function_declaration", "class_declaration", "interface_declaration",
            "type_alias_declaration", "export_statement", "lexical_declaration",
            # Go
            "function_declaration", "method_declaration", "type_declaration",
            # Rust
            "function_item", "struct_item", "enum_item", "impl_item", "trait_item",
            # Java
            "class_declaration", "interface_declaration", "method_declaration",
        }

        for child in tree.root_node.children:
            node_to_check = child

            # If wrapped in an export statement (e.g. `export function ...` in TS/JS), unpack it
            if child.type == "export_statement" and child.children:
                for c in child.children:
                    if c.type in target_types:
                        node_to_check = c
                        break

            if node_to_check.type in target_types:
                s_line = node_to_check.start_point.row + 1
                e_line = node_to_check.end_point.row + 1

                # Extract identifier name using Tree-sitter's standard 'name' field
                name_node = node_to_check.child_by_field_name("name")
                if not name_node and node_to_check.children:
                    for ch in node_to_check.children:
                        if ch.child_by_field_name("name"):
                            name_node = ch.child_by_field_name("name")
                            break
                symbol_name = name_node.text.decode("utf-8") if name_node else "anonymous"

                chunk_content = "\n".join(lines[s_line - 1 : e_line])
                if chunk_content.strip():
                    chunks.append(
                        DocumentChunk.create(
                            source_type=DocumentType.CODE,
                            file_path=file_path,
                            start_line=s_line,
                            end_line=e_line,
                            content=chunk_content,
                            metadata={
                                "language": lang_name,
                                "symbol_name": symbol_name,
                                "symbol_type": node_to_check.type,
                            },
                        )
                    )

        # Fallback if no specific top-level symbols matched: chunk using sliding window
        if not chunks and lines:
            return self._chunk_windowed(content, file_path, language=lang_name)

        return chunks

    def _chunk_windowed(
        self,
        content: str,
        file_path: str,
        language: str = "unknown",
        window_size: int = 60,
        overlap: int = 10,
    ) -> List[DocumentChunk]:
        """Universal sliding-window fallback for unmapped languages, syntax errors, and flat scripts.

        Splits code into 60-line windows with a 10-line overlap. 60 lines (~200-350 tokens)
        safely fits within embedding context limits (512 tokens), preventing silent truncation.
        """
        lines = content.splitlines()
        if not lines:
            return []

        # If file is short enough, emit as a single chunk
        if len(lines) <= window_size:
            return [
                DocumentChunk.create(
                    source_type=DocumentType.CODE,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    content=content,
                    metadata={"fallback": True, "chunk_type": "sliding_window", "language": language},
                )
            ]

        chunks: List[DocumentChunk] = []
        stride = max(1, window_size - overlap)

        for start_idx in range(0, len(lines), stride):
            end_idx = min(len(lines), start_idx + window_size)
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():
                chunks.append(
                    DocumentChunk.create(
                        source_type=DocumentType.CODE,
                        file_path=file_path,
                        start_line=start_idx + 1,
                        end_line=end_idx,
                        content=chunk_content,
                        metadata={
                            "fallback": True,
                            "chunk_type": "sliding_window",
                            "language": language,
                            "window_start": start_idx + 1,
                            "window_end": end_idx,
                        },
                    )
                )

            if end_idx >= len(lines):
                break

        return chunks