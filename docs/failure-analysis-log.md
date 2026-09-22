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

*(Entries will be appended here as we run our Phase 1 through Phase 17 experiments)*
