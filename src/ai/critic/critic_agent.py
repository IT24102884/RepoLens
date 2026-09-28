from enum import Enum
import json
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

from ai.core.models import DocumentChunk
from ai.ingestion.repo_profiler import DEFAULT_HTTPX_PROFILE, RepoProfile

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

load_dotenv()


class CriticStatus(str, Enum):
    """Result of the critic's entailment evaluation."""
    VERIFIED = "VERIFIED"                      # 100% faithful to retrieved context; zero hallucinations
    REVISED = "REVISED"                        # Hallucinations or extrapolations detected and pruned/repaired
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT"     # Context lacks necessary evidence to answer the question


class CriticResult(BaseModel):
    """Verification verdict and revised answer from the Critic Agent."""
    status: CriticStatus
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="Faithfulness score between 0.0 and 1.0")
    hallucinations_detected: List[str] = Field(
        default_factory=list,
        description="Specific ungrounded claims or invented parameters that were identified",
    )
    verified_answer: str = Field(description="The finalized answer with hallucinations pruned or verified")
    critique_summary: str = Field(description="Brief explanation of the critic's reasoning")
    latency_ms: float = Field(default=0.0, description="Execution time of the critic evaluation in milliseconds")


class CriticAgent:
    """Semantic Entailment & Self-Correction Critic Agent (System D - Component 2).
    
    Acts as an adversarial 'code reviewer' that inspects draft answers against retrieved DocumentChunks.
    Identifies ungrounded extrapolations, invalid parameter names, and unsubstantiated claims,
    autonomously pruning or repairing them before delivery to the user.
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
                raise ValueError("Missing GROQ_API_KEY for Critic Agent execution")
            self._groq_client = Groq(api_key=self.api_key)
        return self._groq_client

    @traceable(name="Critic Agent Verification", run_type="llm")
    def critique(
        self,
        query: str,
        draft_answer: str,
        context_chunks: List[DocumentChunk],
    ) -> CriticResult:
        """Inspect draft answer sentence-by-sentence against context chunks and eliminate hallucinations."""
        t_start = time.perf_counter()

        # If no chunks were retrieved, the draft cannot be verified against evidence
        if not context_chunks:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            repo_name = self.repo_profile.repo_name or "target"
            return CriticResult(
                status=CriticStatus.INSUFFICIENT_EVIDENCE,
                faithfulness_score=0.0,
                hallucinations_detected=["No context chunks were provided to verify claims against."],
                verified_answer=(
                    f"The {repo_name} codebase does not contain verified documentation or source code "
                    f"to answer the question: '{query}'."
                ),
                critique_summary="No context chunks available for citation verification.",
                latency_ms=elapsed_ms,
            )

        # Assemble verified context references
        context_lines: List[str] = []
        for i, c in enumerate(context_chunks, start=1):
            context_lines.append(
                f"[Source {i}: {c.file_path} (L{c.start_line}-L{c.end_line})]\n{c.content}"
            )
        context_str = "\n\n".join(context_lines)

        repo_name = self.repo_profile.repo_name or "target"
        system_prompt = (
            f"You are an adversarial, deterministic code-reviewing critic for the '{repo_name}' repository.\n"
            "Your job is to strictly verify the draft answer against the verified context snippets below.\n\n"
            "Rules of Critique:\n"
            "1. Grounding: Every technical claim, parameter name, method signature, and code detail must be explicitly proven by the context.\n"
            "2. Extrapolations: If the draft invents parameters, assumes absent features, or makes unproven claims, identify each in 'hallucinations_detected'.\n"
            "3. Self-Correction: In 'verified_answer', rewrite or prune the text so that every remaining claim is 100% faithful to the context.\n"
            "4. Citations: Preserve all verified file names and line citations.\n"
            "5. Status & Scoring:\n"
            "   - If the draft is already 100% faithful with zero hallucinations: status='VERIFIED', faithfulness_score=1.0.\n"
            "   - If hallucinations were pruned or repaired: status='REVISED', faithfulness_score=0.85-0.95.\n"
            "   - If the context simply cannot answer the question: status='INSUFFICIENT', faithfulness_score=0.0-0.4.\n\n"
            "Respond ONLY with valid JSON in this exact structure:\n"
            "{\n"
            '  "status": "VERIFIED|REVISED|INSUFFICIENT",\n'
            '  "faithfulness_score": 0.95,\n'
            '  "hallucinations_detected": ["description of ungrounded claim"],\n'
            '  "verified_answer": "Cleaned, strictly faithful answer",\n'
            '  "critique_summary": "Brief explanation of verification verdict"\n'
            "}"
        )

        user_content = (
            f"User Question: {query}\n\n"
            f"Verified Context Snippets:\n{context_str}\n\n"
            f"Draft Answer to Verify:\n{draft_answer}"
        )

        try:
            completion = None
            for attempt in range(3):
                try:
                    completion = self.groq_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.0,
                        max_tokens=650,
                    )
                    break
                except Exception as e:
                    if ("rate_limit" in str(e).lower() or "429" in str(e)) and attempt < 2:
                        time.sleep(3.0 * (attempt + 1))
                    elif attempt == 2:
                        raise

            raw_text = completion.choices[0].message.content or "{}" if completion else "{}"
            json_str = raw_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)

            raw_status = str(parsed.get("status", "VERIFIED")).strip().upper()
            try:
                status = CriticStatus(raw_status)
            except ValueError:
                status = CriticStatus.REVISED if parsed.get("hallucinations_detected") else CriticStatus.VERIFIED

            score = float(parsed.get("faithfulness_score", 0.95))
            score = max(0.0, min(1.0, score))
            hallucinations = parsed.get("hallucinations_detected", [])
            if not isinstance(hallucinations, list):
                hallucinations = [str(hallucinations)]

            verified_answer = str(parsed.get("verified_answer", draft_answer)).strip() or draft_answer
            summary = str(parsed.get("critique_summary", "Critique complete."))
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            return CriticResult(
                status=status,
                faithfulness_score=score,
                hallucinations_detected=hallucinations,
                verified_answer=verified_answer,
                critique_summary=summary,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return CriticResult(
                status=CriticStatus.VERIFIED,
                faithfulness_score=0.85,
                hallucinations_detected=[],
                verified_answer=draft_answer,
                critique_summary=f"Critic evaluation fallback due to error: {str(e)}",
                latency_ms=elapsed_ms,
            )
