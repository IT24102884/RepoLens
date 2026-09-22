# Evaluation & Experimentation Plan

## 1. Evaluation Philosophy

In production AI engineering, **we never claim an architectural improvement without empirical, before-and-after proof.** 

Every feature in this project must be evaluated against a strict baseline across defined metrics. We deliberately test failure modes (ambiguity, multi-hop dependencies, conflicting documentation, missing information) rather than only easy cases.

---

## 2. Benchmark Dataset Design

The evaluation suite will contain **40–50 curated benchmark queries** divided into 8 distinct categories:

| Category | Count | Primary Challenge | Expected Sources |
|---|---|---|---|
| **1. Single-Hop Code** | 6 | Precise AST symbol lookup, exact identifier retrieval | `["CODE"]` |
| **2. Single-Hop Docs** | 6 | Architectural definition, configuration guide lookup | `["DOCS"]` |
| **3. Single-Hop Tickets** | 6 | Historical bug triage, regression attribution | `["TICKETS"]` |
| **4. Multi-Hop Code-to-Code** | 6 | Tracing call graph / data flow across files | `["CODE"]` |
| **5. Cross-Source (Code + Docs)** | 6 | Verifying if documented behavior matches implementation | `["CODE", "DOCS"]` |
| **6. Cross-Source (Code + Tickets)** | 6 | Pinpointing which commit/ticket introduced a behavior | `["CODE", "TICKETS"]` |
| **7. Missing / Out-of-Scope** | 6 | Testing graceful refusal (information does not exist) | None / Refusal |
| **8. Ambiguous / Conflicting** | 6 | Testing query reformulation, disambiguation, and critic | Multiple |

### Benchmark Schema (`eval/data/benchmark_schema.json`)
```json
{
  "id": "Q-001",
  "question": "How does the Client dispatch an async HTTP request and where is connection pooling configured?",
  "category": "CROSS_SOURCE",
  "expected_sources": ["CODE", "DOCS"],
  "expected_evidence": [
    "httpx/_client.py:120-165",
    "docs/advanced.md:45-80"
  ],
  "ground_truth_answer": "...",
  "must_refuse": false
}
```

---

## 3. Metrics Specification

### 3.1 Retrieval Metrics
* **Recall@K**: Proportion of ground-truth evidence chunks retrieved in Top-K.
  $$\text{Recall@K} = \frac{|\text{Retrieved}_K \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$
* **Precision@K**: Proportion of retrieved chunks that are relevant.
  $$\text{Precision@K} = \frac{|\text{Retrieved}_K \cap \text{GroundTruth}|}{K}$$
* **Mean Reciprocal Rank (MRR)**: Evaluates how high the first relevant chunk appears in the ranked list.
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

### 3.2 Generation & Grounding Metrics
* **Faithfulness**: Proportion of claims in the generated answer that are directly inferable from the retrieved context (measured via deterministic NLI / LLM-as-judge prompt).
* **Citation Precision**: Proportion of citations in the answer that validly support their preceding claim.
* **Citation Validity**: Percentage of citations whose file path and line bounds strictly match retrieved chunk IDs.
* **Answer Relevance**: Semantic similarity and factual completeness against the ground-truth answer.
* **Refusal Accuracy**: Precision in returning a refusal when `must_refuse == true`.

### 3.3 Agent Orchestration Metrics
* **Routing Accuracy**: Percentage of queries routed to the exact expected sources.
* **Critic Precision & Recall**: Precision and recall of the Critic in identifying unsupported claims vs false alarms.
* **Retry Recovery Rate**: Percentage of queries failing verification on pass 1 that pass after 1 bounded reformulation.

### 3.4 System & Operational Metrics
* **Latency (p50, p95)**: Measured in seconds end-to-end and per graph node.
* **Token Usage**: Total prompt tokens and completion tokens per query.
* **Estimated Cost**: Cost in USD per 1,000 queries.

---

## 4. Controlled Ablation Experiments

To quantify the value of each engineering component, we will run the benchmark across 5 controlled system variants:

| Experiment ID | Configuration | Description |
|---|---|---|
| **System A (Baseline)** | Naive Vector RAG | Fixed-size chunking (500 tokens), dense vector search only, single prompt synthesis, no routing, no critic. |
| **System B** | Hybrid Retrieval | AST-aware chunking + BM25 sparse search + Dense vector search + RRF fusion. |
| **System C** | B + Query Router | Supervisor routes to dedicated Code/Doc/Ticket indexes and emits decomposed subqueries. |
| **System D** | C + Critic | Synthesizer drafts answer with citations; Critic validates citations and claims. |
| **System E (Full)** | D + Bounded Retry | Full state graph with bounded query reformulation (max 2 retries). |

### Master Ablation Results Table (To be populated with real measurements)

| System | Recall@5 | Precision@5 | Faithfulness | Citation Val. | Routing Acc. | P95 Latency | Est. Cost / 1k |
|---|---|---|---|---|---|---|---|
| **System A (Naive RAG)** | *TBD* | *TBD* | *TBD* | *TBD* | N/A | *TBD* | *TBD* |
| **System B (Hybrid)** | *TBD* | *TBD* | *TBD* | *TBD* | N/A | *TBD* | *TBD* |
| **System C (+ Router)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **System D (+ Critic)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **System E (+ Bounded Retry)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 5. Failure Analysis Taxonomy & Log

Every benchmark query that produces an incorrect, ungrounded, or failed response is categorized into our **Failure Analysis Log** (`docs/failure-analysis-log.md`):

### Failure Categories
1. `RETRIEVAL_FAILURE`: Relevant chunk was not retrieved in top-K.
2. `ROUTING_FAILURE`: Query routed to wrong index (e.g. searched Docs when only in Code).
3. `CHUNKING_FAILURE`: AST parser split a critical function or lost parent class context.
4. `SYNTHESIS_FAILURE`: Context contained answer, but LLM failed to infer or synthesize it.
5. `CITATION_FAILURE`: Citation points to wrong file, out-of-bound line numbers, or hallucinated ticket.
6. `CRITIC_FAILURE`: Critic incorrectly rejected a valid answer (False Positive) or accepted a hallucinated answer (False Negative).
7. `HALLUCINATION`: LLM introduced factual claims not present in retrieved context.
8. `REFUSAL_FAILURE`: System guessed on an out-of-scope question instead of refusing.
9. `TIMEOUT_FAILURE`: Graph execution exceeded timeout budget.

### Log Entry Template
```markdown
### [FAIL-001] <Brief Description>
- **Query ID**: Q-012
- **Question**: "..."
- **Expected Behavior**: ...
- **Actual Behavior**: ...
- **Failure Category**: CHUNKING_FAILURE
- **Root Cause**: Method `_send_single_request` was chunked without its enclosing `HTTPTransport` class imports.
- **Hypothesis**: Injecting parent class header and docstring into child method chunks will restore semantic context.
- **Fix**: Updated `ast_chunker.py` to prepend parent class declaration.
- **Before Metric**: Recall@5 = 62.5%
- **After Metric**: Recall@5 = 83.3%
- **Status**: RESOLVED (Regression test added)
```
