import time
from ai.retrieval.chroma_retriever import ChromaRetriever


def main():
    print("\n" + "=" * 60)
    print("REPOLENS CHROMADB VECTOR SEARCH DEMO")
    print("=" * 60)

    # 1. Initialize ChromaRetriever
    retriever = ChromaRetriever()

    # 2. Index the chunks (only computes embeddings if not already in ChromaDB)
    t0 = time.perf_counter()
    retriever.index_chunks(batch_size=50)
    index_time = time.perf_counter() - t0
    print(f"Index readiness verified in {index_time:.2f}s\n")

    # 3. Test Query
    query = "How do connection timeouts work in HTTPX?"
    print(f"Running Search Query: '{query}'")

    t1 = time.perf_counter()
    results = retriever.search(query=query, top_k=3)
    search_latency_ms = (time.perf_counter() - t1) * 1000

    print(f"Retrieval completed in {search_latency_ms:.1f}ms!\n")

    # 4. Display Results
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"--- [RANK {rank}] (Similarity Score: {score:.4f}) " + "-" * 30)
        print(f"Source Type: {chunk.source_type.value}")
        print(f"Location:    {chunk.file_path} (Lines {chunk.start_line} -> {chunk.end_line})")
        print(f"Metadata:    {chunk.metadata}")
        print("-" * 55)
        print("Content Preview:")
        for line in chunk.content.splitlines()[:12]:
            print(f"  {line}")
        if len(chunk.content.splitlines()) > 12:
            print("  ... (truncated for display)")
        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()