# Architecture Decision Records (ADRs)

## Index of Decisions
- [ADR-0001: Separation of Multi-Agent Responsibilities](#adr-0001-separation-of-multi-agent-responsibilities)
- [ADR-0002: AST-Aware Semantic Chunking for Source Code](#adr-0002-ast-aware-semantic-chunking-for-source-code)
- [ADR-0003: Hybrid Retrieval (BM25 + Dense Vector) with Reciprocal Rank Fusion](#adr-0003-hybrid-retrieval-bm25--dense-vector-with-reciprocal-rank-fusion)
- [ADR-0004: Deterministic Pre-Delivery Citation Validation](#adr-0004-deterministic-pre-delivery-citation-validation)
- [ADR-0005: Bounded Self-Correction Loop (MAX_RETRIES = 2) with Graceful Refusal](#adr-0005-bounded-self-correction-loop-max_retries--2-with-graceful-refusal)

---

### ADR-0001: Separation of Multi-Agent Responsibilities
* **Status**: ACCEPTED
* **Context**: In complex engineering QA, single-prompt LLM systems conflate query analysis, information retrieval, synthesis, and self-auditing into one generation step. This leads to prompt bloat, high hallucination rates, and uncontrollable behavior.
* **Decision**: We decouple the pipeline into discrete agent nodes managed by LangGraph:
  1. **Router**: Determines *where* to look without generating answers.
  2. **Retrievers**: Specialized per domain (`Code`, `Docs`, `Tickets`).
  3. **Synthesizer**: Answers *only* with retrieved context and explicit citations.
  4. **Critic/Verifier**: Audits citations and claim support independently.
  5. **Reformulator**: Re-queries *only* when the Critic identifies specific evidence gaps.
* **Consequences**:
  - *Positive*: Independent testability, modular failure localization, measurable routing accuracy.
  - *Trade-off*: Increased orchestration overhead and slightly higher token consumption per multi-hop request.

---

### ADR-0002: AST-Aware Semantic Chunking for Source Code
* **Status**: ACCEPTED
* **Context**: Fixed-size windowing (e.g. 500 tokens with 50-token overlap) arbitrarily splits code mid-function or between class definitions and method implementations, destroying syntax trees and isolating methods from their parent class context.
* **Decision**: We parse Python code into an Abstract Syntax Tree (AST) using Python's native `ast` module. Chunks are bounded by syntactic units (`ClassDef`, `FunctionDef`, `AsyncFunctionDef`). Each chunk retains:
  - Parent class name and docstring.
  - Exact 1-indexed `start_line` and `end_line`.
  - Module-level imports context.
* **Consequences**:
  - *Positive*: Preserves semantic integrity, enables line-accurate citations, improves retrieval recall on function lookups.
  - *Trade-off*: Chunk sizes vary dynamically depending on function length; very long functions need hierarchical inner block chunking.

---

### ADR-0003: Hybrid Retrieval (BM25 + Dense Vector) with Reciprocal Rank Fusion
* **Status**: ACCEPTED
* **Context**: Source code queries heavily rely on exact identifier names (e.g., `TimeoutException`, `HTTPTransport`, `max_keepalive_connections`). Dense vector embeddings frequently struggle with exact string matching and out-of-vocabulary technical tokens.
* **Decision**: We combine BM25 sparse keyword retrieval with dense semantic vector retrieval using Reciprocal Rank Fusion (RRF, $k=60$).
* **Consequences**:
  - *Positive*: Combines semantic concept matching (e.g. "connection pooling") with exact symbol resolution (e.g. `_pool.py`).
  - *Trade-off*: Dual indexing requires maintaining two index structures (in-memory BM25 index + vector store).

---

### ADR-0004: Deterministic Pre-Delivery Citation Validation
* **Status**: ACCEPTED
* **Context**: LLMs can generate plausible-looking citations (e.g. `[src/auth.py:40-60]`) that do not exist in the retrieved context, giving users false confidence in fabricated claims.
* **Decision**: Before delivering an answer to the user, a deterministic Python validator parses all citations using regex, cross-checks the file paths and line ranges against the actual retrieved chunks in the graph state, and rejects or strips unverified citations.
* **Consequences**:
  - *Positive*: Guarantees zero phantom citations reaching the user; deterministic, fast (sub-millisecond), zero LLM cost.
  - *Trade-off*: Requires rigid formatting syntax for citations that LLM prompts must strictly follow.

---

### ADR-0005: Bounded Self-Correction Loop (MAX_RETRIES = 2) with Graceful Refusal
* **Status**: ACCEPTED
* **Context**: Autonomous agent loops without hard bounds can enter infinite cycles, drastically increasing API latency, burning tokens, and ultimately producing ungrounded answers when the requested information simply does not exist.
* **Decision**: We enforce an invariant `MAX_RETRIES = 2`. If the Critic detects insufficient evidence after 2 reformulations, the system halts and executes a `GracefulRefusal` node. The response states:
  1. What evidence was found.
  2. What critical information was missing.
  3. Why it cannot answer further without speculation.
* **Consequences**:
  - *Positive*: Predictable latency ceilings, controlled inference costs, production reliability, zero hallucination on out-of-scope questions.
  - *Trade-off*: Complex edge cases may refuse rather than try a 3rd or 4th attempt.

