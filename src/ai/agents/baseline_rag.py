import os
import time
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from groq import Groq
from ai.core.models import DocumentChunk
from ai.retrieval.chroma_retriever import ChromaRetriever

load_dotenv()


class BaselineRAG:
    """Naive Dense RAG (Ablation 1).
    
    Pulls top chunks from ChromaDB and runs fast LPU generation via Groq.
    Zero complex routing or verification loops for the baseline.
    """

    def __init__(self, model_name: str = "qwen/qwen3.8-27b", top_k: int = 5):
        self.model_name = model_name
        self.top_k = top_k
        self.retriever = ChromaRetriever()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)

    def answer(self, query: str) -> Dict[str, Any]:
        t_start = time.perf_counter()

        # Search Chroma for closest matching chunks
        t_ret = time.perf_counter()
        matches: List[Tuple[DocumentChunk, float]] = self.retriever.search(
            query=query, top_k=self.top_k
        )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # Build readable context references for the LLM
        context_blocks = []
        for i, (chunk, score) in enumerate(matches, start=1):
            context_blocks.append(
                f"[Source {i}: {chunk.file_path} (L{chunk.start_line}-L{chunk.end_line})]\n"
                f"{chunk.content}\n"
            )
        context_str = "\n".join(context_blocks)

        system_instruction = (
            "You are an engineering assistant helping developers navigate a codebase.\n"
            "Answer the question using only the context provided below.\n"
            "Always cite exact file names and line numbers when referencing code or docs.\n"
            "If the context does not contain the answer, explicitly state that you cannot find it.\n\n"
            f"Context:\n{context_str}"
        )

        # Call Groq LPU endpoint
        t_gen = time.perf_counter()
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
        )
        generation_ms = (time.perf_counter() - t_gen) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        answer_text = chat_completion.choices[0].message.content or ""

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": [c.model_dump() for c, _ in matches],
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "total_latency_ms": total_ms,
        }