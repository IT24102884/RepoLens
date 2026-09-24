import json
from pathlib import Path
import time
from typing import Any, Dict, List
from ai.evaluation.metrics import evaluate_keyword_coverage, evaluate_retrieval_recall
from ai.retrieval.chroma_retriever import ChromaRetriever


def compute_reciprocal_rank(retrieved_chunks: List[Dict[str, Any]], expected_files: List[str]) -> float:
    """Calculates reciprocal rank (1/rank) of the first correct file found."""
    if not expected_files:
        return 1.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        file_path = chunk.get("file_path", "").replace("\\", "/")
        if any(exp in file_path for exp in expected_files):
            return 1.0 / rank

    return 0.0


def run_retrieval_benchmark():
    benchmark_file = Path("eval/data/benchmark.jsonl")
    reports_dir = Path("eval/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_file.exists():
        print(f"[!] Missing benchmark dataset at {benchmark_file}")
        return

    # Load test queries
    queries = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print("\n" + "=" * 80)
    print("      REPOLENS RETRIEVAL BENCHMARK — SYSTEM A (CHROMA DENSE VECTORS)")
    print("=" * 80)
    print(f"Evaluating {len(queries)} queries across all knowledge categories...\n")

    retriever = ChromaRetriever()
    results = []

    header = f"{'ID':<6} | {'Category':<18} | {'Hit@1':<6} | {'Recall@5':<9} | {'MRR':<6} | {'Keywords':<9} | {'Latency':<8}"
    print(header)
    print("-" * 80)

    for q in queries:
        qid = q["id"]
        category = q["category"]
        question = q["question"]
        expected_files = q.get("expected_files", [])
        expected_keywords = q.get("expected_keywords", [])

        t0 = time.perf_counter()
        matches = retriever.search(query=question, top_k=5)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_dicts = [c.model_dump() for c, _ in matches]

        recall = evaluate_retrieval_recall(retrieved_dicts, expected_files)
        rr = compute_reciprocal_rank(retrieved_dicts, expected_files)
        kw_cov = evaluate_keyword_coverage(retrieved_dicts, expected_keywords)
        hit_at_1 = "YES" if rr == 1.0 else ("N/A" if not expected_files else "NO")

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "hit_at_1": hit_at_1,
            "recall_at_5": recall,
            "reciprocal_rank": rr,
            "keyword_coverage": kw_cov,
            "latency_ms": latency_ms,
        })

        row = f"{qid:<6} | {category:<18} | {hit_at_1:<6} | {recall*100:>7.1f}% | {rr:>6.2f} | {kw_cov*100:>7.1f}% | {latency_ms:>6.1f}ms"
        print(row)

    print("-" * 80)

    # Calculate overall metrics
    avg_recall = sum(r["recall_at_5"] for r in results) / len(results)
    avg_mrr = sum(r["reciprocal_rank"] for r in results) / len(results)
    avg_kw = sum(r["keyword_coverage"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print("\n" + "=" * 55)
    print("       SYSTEM A RETRIEVAL MASTER SCORECARD")
    print("=" * 55)
    print(f"  Average Recall@5:            {avg_recall * 100:.1f}%")
    print(f"  Mean Reciprocal Rank (MRR):  {avg_mrr:.2f}")
    print(f"  Keyword Coverage:            {avg_kw * 100:.1f}%")
    print(f"  Average Latency:             {avg_latency:.1f} ms")
    print("=" * 55)

    # Save results to json
    report_file = reports_dir / "system_a_retrieval.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "system": "System A (Dense Vector Retrieval via ChromaDB)",
                "summary": {
                    "recall_at_5": avg_recall,
                    "mrr": avg_mrr,
                    "keyword_coverage": avg_kw,
                    "avg_latency_ms": avg_latency,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n[+] Detailed results saved to {report_file}!\n")


if __name__ == "__main__":
    run_retrieval_benchmark()

