import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from ai.agents.routed_rag import RoutedRAG
from ai.core.models import DocumentChunk
from ai.critic.citation_verifier import CitationVerificationReport, CitationVerifier
from ai.critic.critic_agent import CriticAgent, CriticResult, CriticStatus
from ai.ingestion.repo_profiler import DEFAULT_HTTPX_PROFILE, RepoProfile
from ai.retrieval.hybrid_retriever import HybridRetriever
from ai.routing.router import QueryIntent

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

load_dotenv()


class CriticRAG:
    """System D: Self-Correcting Hybrid RAG with Citation & Entailment Critic.
    
    Architecture:
    1. Routed RAG Engine (System C): Fast intent routing, modality filtering, and macro Repo Card generation.
    2. Deterministic Citation Verifier (Component 1): Validates line bounds and repairs drifted coordinates.
    3. Semantic Entailment Critic Agent (Component 2): Audits factual claims, prunes ungrounded extrapolations,
       and assigns a verifiable faithfulness score before delivery to the user.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.8-27b",
        top_k: int = 5,
        rrf_k: int = 60,
        repo_profile: Optional[RepoProfile] = None,
        retriever: Optional[HybridRetriever] = None,
        routed_rag: Optional[RoutedRAG] = None,
        critic_agent: Optional[CriticAgent] = None,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.repo_profile = repo_profile or DEFAULT_HTTPX_PROFILE
        self.routed_rag = routed_rag or RoutedRAG(
            model_name=model_name,
            top_k=top_k,
            rrf_k=rrf_k,
            repo_profile=self.repo_profile,
            retriever=retriever,
        )
        self.citation_verifier = CitationVerifier()
        self.critic_agent = critic_agent or CriticAgent(
            model_name=model_name,
            repo_profile=self.repo_profile,
        )

    @traceable(name="System D (Critic & Verifier RAG)", run_type="chain")
    def answer(self, query: str) -> Dict[str, Any]:
        """Execute System C generation and pass draft through the Self-Correction Critic Gate."""
        t_start = time.perf_counter()

        # Step 1: Run System C (Routing + Modality-Filtered Retrieval + Macro Grounded Generation)
        system_c_output = self.routed_rag.answer(query)

        routing_decision = system_c_output.get("routing_decision", {})
        intent_val = routing_decision.get("intent")
        if intent_val == QueryIntent.OUT_OF_SCOPE.value or intent_val == QueryIntent.OUT_OF_SCOPE:
            # Immediate refusal: Out-of-scope query, no verification needed
            return {
                **system_c_output,
                "system_version": "System D (Self-Correction Critic & Citation Verifier)",
                "critic_status": CriticStatus.VERIFIED.value,
                "faithfulness_score": 1.0,
                "hallucinations_detected": [],
                "critique_summary": "Query deterministically refused as out-of-scope (0 latency).",
                "critic_latency_ms": 0.0,
                "citation_verification": None,
            }

        draft_answer = system_c_output.get("answer", "")
        raw_chunks = system_c_output.get("retrieved_chunks", [])
        retrieved_chunks = [DocumentChunk(**c) for c in raw_chunks]

        # Step 2: Deterministic Citation & Line Coordinate Verification (0ms, 0 tokens)
        citation_report: CitationVerificationReport = self.citation_verifier.verify_and_patch(
            draft_text=draft_answer,
            retrieved_chunks=retrieved_chunks,
        )
        patched_draft = citation_report.corrected_text

        # Step 3: Semantic Entailment & Hallucination Pruning Critic Gate
        t_critic = time.perf_counter()
        critic_result: CriticResult = self.critic_agent.critique(
            query=query,
            draft_answer=patched_draft,
            context_chunks=retrieved_chunks,
        )
        critic_ms = critic_result.latency_ms if critic_result.latency_ms > 0.0 else (time.perf_counter() - t_critic) * 1000
        total_ms = system_c_output.get("total_latency_ms", 0.0) + critic_ms

        # If citation verifier found drifted lines that were patched, record it in critique summary
        summary = critic_result.critique_summary
        hallucinations = list(critic_result.hallucinations_detected)
        if citation_report.drifted_count > 0:
            summary += f" [Repaired {citation_report.drifted_count} drifted line coordinate(s).]"
        if citation_report.hallucinated_count > 0:
            hallucinations.append(
                f"Referenced {citation_report.hallucinated_count} file(s) not in retrieved chunks."
            )

        status_val = critic_result.status.value
        if citation_report.drifted_count > 0 and status_val == CriticStatus.VERIFIED.value:
            status_val = CriticStatus.REVISED.value

        return {
            "query": query,
            "draft_answer": draft_answer,
            "answer": critic_result.verified_answer,
            "retrieved_chunks": raw_chunks,
            "retrieval_latency_ms": system_c_output.get("retrieval_latency_ms", 0.0),
            "generation_latency_ms": system_c_output.get("generation_latency_ms", 0.0),
            "critic_latency_ms": round(critic_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "system_version": "System D (Self-Correction Critic & Citation Verifier)",
            "routing_decision": routing_decision,
            "critic_status": status_val,
            "faithfulness_score": round(critic_result.faithfulness_score, 2),
            "hallucinations_detected": hallucinations,
            "critique_summary": summary,
            "citation_verification": citation_report.model_dump(),
        }
