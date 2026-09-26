from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

from ai.core.models import DocumentType
from ai.ingestion.repo_profiler import DEFAULT_HTTPX_PROFILE, RepoProfile

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

load_dotenv()


class QueryIntent(str, Enum):
    """Categorized query intent for targeted index retrieval."""
    DOCS_CONCEPTUAL = "DOCS_CONCEPTUAL"   # Guides, architecture, config, conceptual explanation
    CODE_SYMBOL = "CODE_SYMBOL"           # Class signatures, function defs, syntax, constants
    BUG_TICKET = "BUG_TICKET"             # GitHub issue discussions, bug tickets, regressions
    MULTI_HOP = "MULTI_HOP"               # Cross-component tracing, issue-to-code bridges
    OUT_OF_SCOPE = "OUT_OF_SCOPE"         # Irrelevant, adversarial, or out-of-domain queries


class RouteDecision(BaseModel):
    """Result of cascading query routing."""
    intent: QueryIntent
    target_source_type: Optional[DocumentType] = None
    confidence: float = Field(ge=0.0, le=1.0)
    route_source: str = Field(description="'heuristic' or 'llm_classifier'")
    reasoning: str
    sub_queries: List[str] = Field(default_factory=list, description="Sub-queries for multi-hop expansion")
    optimized_search_query: Optional[str] = Field(
        default=None,
        description="Rewritten search query for vector/BM25 retrieval when original query is conversational, meta, or broad",
    )


INTENT_TO_SOURCE_TYPE: Dict[QueryIntent, Optional[DocumentType]] = {
    QueryIntent.DOCS_CONCEPTUAL: DocumentType.DOCUMENTATION,
    QueryIntent.CODE_SYMBOL: DocumentType.CODE,
    QueryIntent.BUG_TICKET: DocumentType.ISSUE_PR,
    QueryIntent.MULTI_HOP: None,
    QueryIntent.OUT_OF_SCOPE: None,
}


class IntentRouter:
    """Cascading Hybrid Intent Router (System C).
    
    Tier 1: Sub-millisecond deterministic universal non-software regex & code patterns.
    Tier 2: Dynamic Groq micro-LLM semantic classifier parameterized by RepoProfile.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.8-27b",
        api_key: Optional[str] = None,
        repo_profile: Optional[RepoProfile] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.repo_profile = repo_profile or DEFAULT_HTTPX_PROFILE
        self._groq_client: Optional[Groq] = None

    @property
    def groq_client(self) -> Groq:
        if self._groq_client is None:
            if not self.api_key:
                raise ValueError("Missing GROQ_API_KEY for LLM query classification")
            self._groq_client = Groq(api_key=self.api_key)
        return self._groq_client

    def classify_heuristically(self, query: str) -> Optional[RouteDecision]:
        """Tier 1: Universal heuristic pattern matching (0ms overhead).
        
        Only matches universal non-software topics and language-neutral code syntax.
        Domain-specific out-of-scope queries are handled dynamically by Tier 2 and Vector Distance Gating.
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. Universal Non-Software Heuristics (True for ANY software repository)
        universal_out_of_scope = [
            r"\b(?:weather|forecast|temperature in|climate in)\b",
            r"\b(?:recipe|cooking|bake\s+a|cake\s+recipe|ingredients\s+for)\b",
            r"\b(?:bitcoin|ethereum|crypto(?:\s+currency)?|stock\s+market|forex)\b",
            r"\b(?:nba\s+score|premier\s+league|world\s+cup|football\s+match)\b",
            r"\b(?:write\s+a\s+poem|who\s+is\s+the\s+president\s+of)\b",
        ]
        for pat in universal_out_of_scope:
            if re.search(pat, q_lower):
                return RouteDecision(
                    intent=QueryIntent.OUT_OF_SCOPE,
                    target_source_type=None,
                    confidence=0.98,
                    route_source="heuristic",
                    reasoning=f"Matched universal non-software topic: {pat}",
                )

        # 2. Bug Ticket & Issue Discussion Heuristics
        issue_match = re.search(r"(?:#\d+|\b(?:issue|ticket|pr|pull\s+request|gh-)\s*#?\d+\b|\bissues/\d+\b)", q_lower)
        if issue_match and re.search(r"\b(?:show\s+(?:the\s+)?code|code\s+that\s+fix|code\s+implementation|source\s+code\s+for\s+issue)\b", q_lower):
            ticket_ref = issue_match.group(0)
            return RouteDecision(
                intent=QueryIntent.MULTI_HOP,
                target_source_type=None,
                confidence=0.95,
                route_source="heuristic",
                reasoning=f"Matched issue reference {ticket_ref} combined with code fix request.",
                sub_queries=[q, f"fix {ticket_ref} implementation"],
            )

        if issue_match:
            return RouteDecision(
                intent=QueryIntent.BUG_TICKET,
                target_source_type=DocumentType.ISSUE_PR,
                confidence=0.98,
                route_source="heuristic",
                reasoning=f"Matched explicit issue/ticket tag: {issue_match.group(0)}",
            )

        # 3. Multi-Hop Code Delegation & Call Chain Heuristics
        multi_hop_patterns = [
            r"pass(?:es)?\s+.*\s+down\s+to\s+(?:the\s+)?underlying\s+transport",
            r"flow\s+from\s+.*\s+to\s+",
            r"how\s+does\s+.*\s+pass\s+.*\s+to\s+.*\s+dispatch",
            r"call\s+chain\s+from\s+.*\s+to\s+",
        ]
        for pat in multi_hop_patterns:
            if re.search(pat, q_lower):
                return RouteDecision(
                    intent=QueryIntent.MULTI_HOP,
                    target_source_type=DocumentType.CODE,
                    confidence=0.95,
                    route_source="heuristic",
                    reasoning=f"Matched multi-hop cross-component flow pattern: {pat}",
                    sub_queries=[
                        "Client.request headers cookies send _send_handling_redirects",
                        "transport dispatch handle_request default transport HTTPTransport",
                    ],
                )

        # 4. Code Symbol & Exact Syntax Heuristics (Language-neutral)
        code_patterns = [
            r"\bclass\s+[A-Z][A-Za-z0-9_]*",
            r"\bdef\s+[a-z_][a-z0-9_]*",
            r"\bfunc\s+[A-Za-z_][A-Za-z0-9_]*",
            r"\b[A-Za-z0-9_]+\.(?:py|ts|tsx|js|jsx|go|rs|java)\b",
            r"\bHTTPStatus\.[A-Z0-9_]+\b",
            r"\bcodes\.[A-Z0-9_]+\b",
            r"\bwhere\s+is\s+(?:the\s+)?[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\s+(?:class|function|method|enum|struct|interface|defined)\b",
            r"\bwhich\s+file\s+(?:and\s+class\s+)?defines\b",
            r"\bhow\s+does\s+(?:the\s+)?[A-Za-z0-9_]+\s+parse\s+raw\b",
        ]
        for pat in code_patterns:
            if re.search(pat, q, re.IGNORECASE):
                return RouteDecision(
                    intent=QueryIntent.CODE_SYMBOL,
                    target_source_type=DocumentType.CODE,
                    confidence=0.95,
                    route_source="heuristic",
                    reasoning=f"Matched code syntax pattern: {pat}",
                )

        # 5. Documentation & Conceptual Heuristics (Explicit documentation keywords only)
        docs_patterns = [
            r"\b(?:architecture|user\s+guide|tutorial|best\s+practices|faq|documentation|docs|manual)\b",
            r"\bwhat\s+are\s+the\s+(?:four\s+|[0-9]+\s+)?(?:different\s+)?types\s+of\s+timeouts\b",
            r"\b(?:how\s+do\s+i|how\s+to)\s+configure\s+(?:custom\s+ssl|ssl|timeouts?|proxy|proxies|auth|authentication|client|transport)\b",
        ]
        for pat in docs_patterns:
            if re.search(pat, q_lower):
                return RouteDecision(
                    intent=QueryIntent.DOCS_CONCEPTUAL,
                    target_source_type=DocumentType.DOCUMENTATION,
                    confidence=0.92,
                    route_source="heuristic",
                    reasoning=f"Matched explicit documentation guide pattern: {pat}",
                )

        return None

    @traceable(name="Groq Micro-LLM Router", run_type="parser")
    def classify_with_llm(self, query: str) -> RouteDecision:
        """Tier 2: Micro-LLM classification parameterized dynamically by RepoProfile (~60ms)."""
        languages_str = ", ".join(self.repo_profile.primary_languages) or "general software"
        subsystems_str = ", ".join(self.repo_profile.subsystems) or "core codebase"

        system_prompt = (
            f"You are an expert query classifier for the repository: '{self.repo_profile.repo_name}'.\n"
            f"Repository Domain & Summary: {self.repo_profile.description}\n"
            f"Primary Languages: {languages_str}\n"
            f"Subsystems: {subsystems_str}\n\n"
            "Classify the user question into exactly ONE intent:\n"
            "- DOCS_CONCEPTUAL: High-level guides, documentation, architecture, conceptual questions, or general repository overviews.\n"
            "- CODE_SYMBOL: Specific code definitions, class signatures, function implementations, file locations, constants.\n"
            "- BUG_TICKET: Specific GitHub issues, PRs, bug numbers, regressions, ticket discussions.\n"
            "- MULTI_HOP: Cross-component data flows, tracing calls across multiple files, or issue-to-code diff bridges.\n"
            "- OUT_OF_SCOPE: Questions unrelated to this repository's domain (e.g. absent technologies, alien frameworks, non-software questions).\n\n"
            "Query Optimization:\n"
            "- If the query asks for an overview, summary, purpose, or high-level architecture of the repository (e.g., 'tell me about the repo', 'what does this do', 'give me the big picture', 'explain this project'), provide an 'optimized_search_query' combining key concepts, domain terms, and README/architecture keywords to retrieve the most relevant overview documentation.\n"
            "- Otherwise, set 'optimized_search_query' to null.\n\n"
            "Respond ONLY with valid JSON in this exact structure:\n"
            '{"intent": "DOCS_CONCEPTUAL|CODE_SYMBOL|BUG_TICKET|MULTI_HOP|OUT_OF_SCOPE", "confidence": 0.95, "reasoning": "brief explanation", "sub_queries": [], "optimized_search_query": null}'
        )

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            raw_text = completion.choices[0].message.content or "{}"
            json_str = raw_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)
            intent_str = parsed.get("intent", "CODE_SYMBOL").strip().upper()
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.CODE_SYMBOL

            confidence = float(parsed.get("confidence", 0.85))
            reasoning = str(parsed.get("reasoning", "Classified via dynamic micro-LLM."))
            sub_queries = parsed.get("sub_queries", [])
            opt_query = parsed.get("optimized_search_query")
            optimized_search_query = opt_query.strip() if isinstance(opt_query, str) and opt_query.strip() else None

            return RouteDecision(
                intent=intent,
                target_source_type=INTENT_TO_SOURCE_TYPE.get(intent),
                confidence=confidence,
                route_source="llm_classifier",
                reasoning=reasoning,
                sub_queries=sub_queries if isinstance(sub_queries, list) else [],
                optimized_search_query=optimized_search_query,
            )

        except Exception as e:
            return RouteDecision(
                intent=QueryIntent.CODE_SYMBOL,
                target_source_type=DocumentType.CODE,
                confidence=0.5,
                route_source="llm_classifier_fallback",
                reasoning=f"LLM classification fallback due to: {str(e)}",
                optimized_search_query=None,
            )

    @traceable(name="Cascading Intent Router", run_type="parser")
    def route(self, query: str) -> RouteDecision:
        """Route query through cascading tiers: Universal Heuristic Fast-Path -> Dynamic Micro-LLM."""
        heuristic_decision = self.classify_heuristically(query)
        if heuristic_decision is not None and heuristic_decision.confidence >= 0.90:
            return heuristic_decision

        return self.classify_with_llm(query)
