# Failure Analysis Log

This log records every significant failure encountered during benchmark runs, integration tests, and ablation experiments. We follow the principle: **Never hide failures; systematically diagnose, hypothesize, fix, verify, and add regression tests.**

---

## Failure Schema Template

```markdown
### [FAIL-XXX] <Concise failure description>
- **Date**: YYYY-MM-DD
- **Query ID**: Q-XXX
- **Question**: "<Exact user/benchmark query>"
- **Expected Behavior**: <What the system should have retrieved, routed, or answered>
- **Actual Behavior**: <What the system actually did>
- **Failure Category**: RETRIEVAL_FAILURE | ROUTING_FAILURE | CHUNKING_FAILURE | RERANKING_FAILURE | SYNTHESIS_FAILURE | CITATION_FAILURE | CRITIC_FAILURE | HALLUCINATION | REFUSAL_FAILURE | TIMEOUT | COST_FAILURE
- **Root Cause**: <In-depth technical explanation of why the failure occurred>
- **Hypothesis**: <Proposed single-variable architectural or algorithmic change to fix it>
- **Fix**: <Specific code/prompt/configuration change applied>
- **Before Metric**: <Metric before the fix>
- **After Metric**: <Metric after the fix>
- **Regression Test**: <Path to automated test covering this case>
- **Status**: OPEN | IN_PROGRESS | RESOLVED | WONT_FIX (with rationale)
```

---

## Log Entries

### [FAIL-001] Q-002: Ungrounded Extrapolation & Low Faithfulness (0.0%) on Code Definition
- **Date**: 2026-09-23
- **Query ID**: Q-002
- **Question**: "Where is the URL class defined and how does it parse raw byte paths?"
- **Expected Behavior**: Retrieve `httpx/_urls.py` chunk containing the `class URL` declaration and constructor/parser logic, accurately citing where `URL` is instantiated.
- **Actual Behavior**: Dense Chroma search retrieved only property accessors in `_urls.py` (L280-L295) and module docstrings in `_urlparse.py` (L1-L17). The generator LLM extrapolated and made assertions not backed by the retrieved text, resulting in a **0.0% Faithfulness score** from the LLM judge.
- **Failure Category**: RETRIEVAL_FAILURE & HALLUCINATION
- **Root Cause**: Dense vector embeddings match general semantic concepts ("URL parsing", "byte paths") rather than exact symbol definitions (`class URL:`), causing critical definition context to be missed.
- **Hypothesis**: Sparse keyword search (BM25) with exact token weighting on `"class URL"` combined with Reciprocal Rank Fusion (System B) will bring the class declaration into the top-3.
- **Fix**: Implement BM25 retriever + Hybrid RRF retrieval in Ablation 2.
- **Before Metric**: Recall@5: 100.0% (retrieved related files), Faithfulness: 0.0%.
- **After Metric**: *Pending System B evaluation*
- **Regression Test**: `tests/evaluation/test_system_a_regressions.py`
- **Status**: OPEN

---

### [FAIL-002] Q-003 & Q-008: Dense Semantic Embedding Blindspot on Bug Tickets (0.0% Recall)
- **Date**: 2026-09-23
- **Query ID**: Q-003, Q-008
- **Question**: "Why does AsyncClient hang when calling client.stream without an async with context manager according to issue 1240?" / "How was the HTTP/2 keepalive connection drop issue fixed in ticket 1405?"
- **Expected Behavior**: Retrieve `issues/1240` and `issues/1405` containing discussions on connection pool exhaustion and keepalive drops.
- **Actual Behavior**: Recall@5 dropped to **0.0%** for both ticket queries. Dense embeddings favored generic documentation over issue descriptions.
- **Failure Category**: RETRIEVAL_FAILURE
- **Root Cause**: `all-MiniLM-L6-v2` dense embeddings project conversational/bug language differently from formal documentation, and without query routing or exact keyword boosting on `"issue 1240"`, ticket chunks get buried under generic doc chunks.
- **Hypothesis**: BM25 keyword indexing directly rewards exact token matches like `"1240"` or `"issue 1240"`, guaranteeing high rank in the candidate pool.
- **Fix**: Integrate BM25 index over code, docs, and tickets with RRF.
- **Before Metric**: Recall@5: 0.0%.
- **After Metric**: *Pending System B evaluation*
- **Status**: OPEN

---

### [FAIL-003] Q-006: Multi-Hop Disconnect in Dense-Only Single-Shot Retrieval (0.0% Recall)
- **Date**: 2026-09-23
- **Query ID**: Q-006
- **Question**: "How does Client.request pass headers and cookies down to the underlying transport dispatch?"
- **Expected Behavior**: Retrieve both the client-level dispatch in `httpx/_client.py` and the transport handler in `httpx/_transports/default.py`.
- **Actual Behavior**: Single flat vector search retrieved partial changelog notes and transport overview docs, completely missing the actual code pathways. Recall: 0.0%, Refusal: 0.0% (system admitted lack of context).
- **Failure Category**: RETRIEVAL_FAILURE
- **Root Cause**: Answering multi-hop architectural flow questions requires cross-file linking (`_client.py` $\rightarrow$ `_transports/default.py`). A naive flat top-5 retrieval cannot bridge multi-file call chains.
- **Hypothesis**: Query decomposition and multi-agent routing (System C & D) will split the question into hop-1 (client request handling) and hop-2 (transport dispatch).
- **Fix**: Multi-agent router & iterative query expansion.
- **Before Metric**: Recall@5: 0.0%.
- **After Metric**: *Pending System C/D evaluation*
- **Status**: OPEN

