import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq

from ai.core.models import DocumentChunk, DocumentType
from ai.ingestion.repo_profiler import DEFAULT_HTTPX_PROFILE, RepoProfile
from ai.retrieval.hybrid_retriever import HybridRetriever
from ai.routing.router import IntentRouter, QueryIntent, RouteDecision

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

load_dotenv()


class RoutedRAG:
    """System C: Intent-Routed & Modality-Filtered Hybrid RAG.
    
    Architecture:
    1. Cascading Router: Tier 1 regex fast-path (0ms) -> Tier 2 Groq micro-LLM (~60ms).
    2. Modality Filtering: Targets retrieval specifically to Code, Docs, or Bug Tickets.
    3. Multi-Hop Decomposition: Decomposes cross-component queries into focused sub-queries.
    4. Deterministic Refusal: Instantly rejects out-of-scope queries with 0 latency & 0 hallucination.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.8-27b",
        top_k: int = 5,
        rrf_k: int = 60,
        repo_profile: Optional[RepoProfile] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.repo_profile = repo_profile or DEFAULT_HTTPX_PROFILE
        self.router = IntentRouter(model_name=model_name, repo_profile=self.repo_profile)
        self.retriever = retriever or HybridRetriever(rrf_k=rrf_k)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)

    @traceable(name="Groq LPU Generation", run_type="llm")
    def _call_llm(self, system_instruction: str, query: str) -> str:
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content or ""

    def _retrieve_multihop(self, query: str, decision: RouteDecision) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve candidates across multiple sub-queries for multi-hop questions."""
        sub_queries = decision.sub_queries if decision.sub_queries else [query]
        candidates_per_subquery = max(2, self.top_k // len(sub_queries) + 1)

        combined_results: List[Tuple[DocumentChunk, float]] = []
        seen_ids = set()

        for sq in sub_queries:
            sub_matches = self.retriever.search(
                query=sq,
                top_k=candidates_per_subquery,
                filter_type=decision.target_source_type,
            )
            for chunk, score in sub_matches:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    combined_results.append((chunk, score))

        return combined_results[: self.top_k]

    @traceable(name="System C (Routed Hybrid RAG)", run_type="chain")
    def answer(self, query: str) -> Dict[str, Any]:
        t_start = time.perf_counter()

        # Step 1: Cascading Intent Routing
        decision: RouteDecision = self.router.route(query)

        # Step 2: Instant Refusal for Out-of-Scope Queries (0 latency, 0 hallucination)
        if decision.intent == QueryIntent.OUT_OF_SCOPE:
            total_ms = (time.perf_counter() - t_start) * 1000
            refusal_msg = (
                f"I am an engineering assistant specialized exclusively in the encode/httpx repository. "
                f"The question '{query}' is outside the scope of this repository (no relevant documentation, "
                f"source code, or issue tickets found). Therefore, I cannot provide an answer based on this codebase."
            )
            return {
                "query": query,
                "answer": refusal_msg,
                "retrieved_chunks": [],
                "retrieval_latency_ms": 0.0,
                "generation_latency_ms": 0.0,
                "total_latency_ms": total_ms,
                "system_version": "System C (Cascading Intent Router)",
                "routing_decision": decision.model_dump(),
            }

        # Step 3: Modality-Filtered Retrieval
        t_ret = time.perf_counter()
        if decision.intent == QueryIntent.MULTI_HOP and decision.sub_queries:
            matches = self._retrieve_multihop(query, decision)
        else:
            matches = self.retriever.search(
                query=query,
                top_k=self.top_k,
                filter_type=decision.target_source_type,
            )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # Step 4: Context Assembly
        context_blocks = []
        for i, (chunk, score) in enumerate(matches, start=1):
            context_blocks.append(
                f"[Source {i}: {chunk.file_path} (L{chunk.start_line}-L{chunk.end_line})]\n"
                f"{chunk.content}\n"
            )
        context_str = "\n".join(context_blocks)

        # Step 5: Grounded LLM Generation Prompt
        system_instruction = (
            "You are an engineering assistant helping developers navigate the encode/httpx codebase.\n"
            f"Query Intent: {decision.intent.value} (Route: {decision.route_source}).\n"
            "Answer the user's question using only the verified context provided below.\n"
            "Always cite exact file names and line numbers when referencing code or documentation.\n"
            "If the context does not contain the answer, explicitly state that you cannot find it.\n\n"
            f"Context:\n{context_str}"
        )

        t_gen = time.perf_counter()
        answer_text = self._call_llm(system_instruction, query)
        generation_ms = (time.perf_counter() - t_gen) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": [c.model_dump() for c, _ in matches],
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "total_latency_ms": total_ms,
            "system_version": "System C (Cascading Intent Router)",
            "routing_decision": decision.model_dump(),
        }
