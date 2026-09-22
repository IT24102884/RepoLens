# System Requirements Specification

## 1. Functional Requirements (FR)

### FR-1: Multi-Source Ingestion
* **FR-1.1**: The system must ingest Git repositories containing Python code, Markdown/RST documentation, and JSON-exported issues/tickets.
* **FR-1.2**: Binary files, build artifacts (`dist/`, `build/`), virtual environments (`.venv/`), and hidden VCS folders (`.git/`) must be automatically excluded.
* **FR-1.3**: The system must compute SHA-256 hashes for all ingested files to support incremental indexing (only re-indexing modified or newly created files).

### FR-2: AST-Aware Semantic Chunking
* **FR-2.1**: Code chunking must parse Python AST to identify semantic boundaries (Classes, Functions, Async Functions, Module-level blocks).
* **FR-2.2**: Each code chunk must preserve 1-indexed `start_line` and `end_line`, the file path, and symbol hierarchy (e.g. `ClassName.method_name`).
* **FR-2.3**: Documentation must be parsed into hierarchical sections preserving header breadcrumbs (e.g. `Installation > Prerequisites`).

### FR-3: Hybrid Retrieval (Dense + Sparse)
* **FR-3.1**: The system must maintain both a dense vector index (semantic embeddings) and a sparse BM25 index (exact token matches).
* **FR-3.2**: Results from both dense and sparse retrievers must be merged using Reciprocal Rank Fusion (RRF) with configurable parameter $k=60$.
* **FR-3.3**: Retrieval must support filtering by `source_type` (`CODE`, `DOCS`, `TICKETS`).

### FR-4: Query Routing & Decomposition
* **FR-4.1**: The Supervisor/Router must classify incoming questions into required target sources (`CODE`, `DOCS`, `TICKETS`).
* **FR-4.2**: The Router must classify query complexity (`SINGLE_HOP` vs `MULTI_HOP`).
* **FR-4.3**: For multi-hop questions, the Router must produce targeted subqueries per source domain.

### FR-5: Parallel Retrieval Execution
* **FR-5.1**: Retrieval across multiple target sources must execute concurrently via asynchronous tasks.
* **FR-5.2**: Each retrieval task must enforce a timeout (e.g. 3.0s) and handle failures gracefully without crashing the overall graph.

### FR-6: Grounded Synthesis with Strict Citations
* **FR-6.1**: The Synthesizer must generate answers exclusively using retrieved context chunks.
* **FR-6.2**: Every factual claim must include an inline citation in the standardized format `[path/to/file:start-end]` or `[TICKET-id]`.
* **FR-6.3**: If the retrieved evidence is insufficient to answer the query, the synthesizer must explicitly state what information is missing.

### FR-7: Critic & Verification Stage
* **FR-7.1**: The Critic agent must review the draft answer against the retrieved evidence.
* **FR-7.2**: The Critic must verify that every citation points to an actual chunk in the retrieved context and that the text supports the claim.
* **FR-7.3**: The Critic must return a structured JSON response with verdict (`SUFFICIENT` or `INSUFFICIENT`), list of unsupported claims, and a reformulated query for missing information.

### FR-8: Bounded Self-Correction Loop
* **FR-8.1**: If the Critic returns `INSUFFICIENT` and `retry_count < 2`, the system must execute the reformulated query.
* **FR-8.2**: The system must enforce a hard bound of `MAX_RETRIES = 2`.
* **FR-8.3**: If the limit is reached without full verification, the system must output a graceful refusal detailing what was confirmed and what remained unverified.

### FR-9: Observability & Tracing
* **FR-9.1**: Every query execution must generate a unique `trace_id`.
* **FR-9.2**: Telemetry must log: query, router decision, subqueries, retrieved chunk IDs, LLM prompt tokens, completion tokens, latency per node, critic verdicts, and retry counts.

### FR-10: API & Verification Harness
* **FR-10.1**: A FastAPI backend must expose endpoints for `/query`, `/health`, and `/index`.
* **FR-10.2**: An evaluation CLI must run benchmark datasets and output comparative metrics.

---

## 2. Non-Functional Requirements (NFR)

### NFR-1: Faithfulness & Hallucination Prevention
* The system must achieve a Faithfulness metric $\ge 95\%$ on benchmark evaluation. Fabricated code symbols, ungrounded ticket IDs, or imaginary functions must be strictly eliminated by the Critic.

### NFR-2: Citation Validity
* 100% of generated citations must be deterministically verifiable against the retrieved context IDs and line bounds.

### NFR-3: Latency Budgets
* P50 end-to-end query latency for single-hop queries $\le 5\text{s}$.
* P95 latency across multi-hop queries with 1 retry $\le 12\text{s}$.
* Ingestion processing throughput $\ge 200$ files/minute.

### NFR-4: Determinism & Testability
* Every core component (File Filter, AST Chunker, BM25 Indexer, Citation Validator, State Machine Transitions) must have $\ge 85\%$ unit test coverage.
* Graph execution paths must be mockable without live network calls.

### NFR-5: Incremental Indexing Efficiency
* Incremental re-indexing of a repository with $< 5\%$ modified files must complete in $\le 10\%$ of the time of a full clean index.

### NFR-6: Modularity & Vendor Independence
* Storage backends (Vector DB, BM25) and LLM providers must be abstracted behind Python Protocols (`abc.ABC`), enabling seamless swapping between OpenAI, Gemini, Anthropic, or local Ollama instances.

