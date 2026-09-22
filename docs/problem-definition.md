# Problem Definition & System Scope

## 1. The Core Problem

In modern software engineering organizations, knowledge is fragmented across multiple disparate systems:
1. **Source Code**: The definitive source of operational truth, classes, interfaces, call-graphs, and data flows.
2. **Project Documentation**: Architectural decision records (ADRs), guides, setup manuals, and design specifications.
3. **Engineering Tickets / Issues**: Bug reports, PR discussions, historical context, decisions, and regression post-mortems.

When engineers join a team or investigate complex cross-cutting concerns, answering questions like:
- *"Where is authentication implemented and how does it interact with the database session?"*
- *"Is the behavior documented in docs/api.md actually implemented in the codebase?"*
- *"Why was this architectural change introduced in issue #142?"*

requires manual multi-hop navigation across code, markdown, and tickets. Standard search tools (e.g., keyword search, GitHub search) fail on semantic nuance, while naive RAG systems suffer from:
1. **Source Mismatch (Routing Failure)**: Searching documentation when the answer only exists in code, or searching code when the rationale only exists in a ticket.
2. **Naive Chunking Loss**: Fixed-size windowing cuts off classes, function bodies, or AST syntax context, leaving retrievers with fragmented and un-parseable code snippets.
3. **Hallucination & Speculation**: Standard LLMs invent functions, parameters, or ticket details when context is incomplete.
4. **Lack of Evidence Traceability**: Answers without verifiable line ranges or ticket IDs are unacceptable for engineering auditing.

---

## 2. System Boundaries

### What the System IS (In Scope)
* An evidence-backed, self-correcting engineering knowledge assistant.
* A multi-agent system orchestrating source routing, hybrid retrieval (BM25 + Dense embeddings), AST-aware chunking, evidence fusion, synthesis, and critic verification.
* A deterministic citation validator ensuring every claim is backed by exact file paths, line ranges, or ticket IDs.
* A bounded retry mechanism (`MAX_RETRIES = 2`) that reformulates queries when retrieved evidence is insufficient.
* An empirical evaluation framework measuring retrieval recall/precision, answer faithfulness, citation accuracy, routing accuracy, and latency/cost.

### What the System IS NOT (Out of Scope / Non-Goals)
* **Not an autonomous code-generation bot**: It does not write pull requests or refactor repositories autonomously.
* **Not a live IDE autocomplete engine**: It is designed for deep architectural, multi-hop, and provenance questions, not real-time 50ms keystroke completion.
* **Not a generic web chat**: It answers exclusively from the indexed repository, documentation, and engineering tickets. If evidence is absent, it must refuse and state what is missing.

---

## 3. Supported Sources & Ingestion Boundaries

| Source Type | Input Format | Extracted Representation | Key Metadata |
|---|---|---|---|
| **Source Code** | `.py` (expandable to `.ts`, `.go`) | AST-aware syntax blocks (Classes, Methods, Module functions) | File path, symbol name, start_line, end_line, parent class, docstrings, imports |
| **Documentation** | `.md`, `.rst`, `.txt` | Hierarchical sections bounded by Markdown headings | File path, heading path (`H1 > H2`), section line ranges |
| **Engineering Tickets** | JSON / GitHub Issues API export | Issue/Ticket objects with titles, descriptions, comments, status | Ticket ID, title, author, labels, state, timestamps, linked files/commits |

---

## 4. Expected Users & Personas

1. **Staff / Senior Architect**: Auditing whether the implemented code adheres to architectural documentation and security specifications.
2. **New Software Engineer / Onboarding Developer**: Exploring the call flows, understanding where core abstractions live, and discovering historical context for quirks.
3. **DevOps / SRE / Quality Engineer**: Triaging regressions, identifying which PR or ticket modified a particular behavior, and finding missing edge cases.

---

## 5. Success Criteria

1. **Evidence Grounding**: Zero unsupported claims tolerated in synthesized answers (100% citation verification rate against retrieved chunks).
2. **Graceful Refusal**: When evidence is missing or ambiguous, the system must explicitly state the deficit rather than guessing.
3. **Routing Precision**: $\ge 90\%$ accuracy in dispatching queries to the correct source subsets (Code, Docs, Tickets).
4. **Retrieval Recall**: Top-K retrieval recall $\ge 85\%$ across complex multi-hop queries using hybrid search.
5. **Bounded Latency & Cost**: Bounded retry loop (`MAX_RETRIES = 2`) with an end-to-end p95 query latency budget $\le 12\text{s}$ (local/fast model).
6. **Measurable Ablation**: Documented before-and-after empirical impact of each architectural stage (Basic RAG $\rightarrow$ Hybrid $\rightarrow$ Router $\rightarrow$ Critic $\rightarrow$ Retry).

