import sys
from ai.agents.baseline_rag import BaselineRAG


def main():
    rag = BaselineRAG()

    # Fallback to default query if none provided in CLI
    query = (
        "How do I configure custom timeouts in HTTPX, and what are the 4 types of timeouts supported?"
    )
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print("\n" + "=" * 65)
    print(f"QUESTION: {query}")
    print("=" * 65 + "\n")

    # Run the full retrieval + LLM synthesis
    result = rag.answer(query)

    print("ANSWER:")
    print("-" * 65)
    print(result["answer"])
    print("-" * 65)

    print("\nPERFORMANCE:")
    print(f"  - Retrieval:  {result['retrieval_latency_ms']:.1f} ms  (ChromaDB Top-5)")
    print(f"  - Generation: {result['generation_latency_ms']:.1f} ms (Groq Qwen 27B)")
    print(f"  - Total:      {result['total_latency_ms']:.1f} ms ({result['total_latency_ms']/1000:.2f}s)\n")


if __name__ == "__main__":
    main()

