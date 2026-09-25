import argparse
import json
from pathlib import Path
import time
from ai.agents.baseline_rag import BaselineRAG
from ai.agents.hybrid_rag import HybridRAG
from ai.agents.routed_rag import RoutedRAG
from ai.evaluation.metrics import (
    evaluate_citation_presence,
    evaluate_faithfulness,
    evaluate_keyword_coverage,
    evaluate_refusal_accuracy,
    evaluate_retrieval_recall,
)


def run_benchmark(system_choice: str = "c"):
    benchmark_file = Path("eval/data/benchmark.jsonl")
    reports_dir = Path("eval/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_file.exists():
        print(f"Benchmark file not found at {benchmark_file}")
        return

    # Load test questions
    questions = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    sys_key = system_choice.strip().lower()
    if sys_key == "a":
        rag = BaselineRAG()
        sys_name = "SYSTEM A (NAIVE DENSE RAG)"
        report_file = "system_a_baseline.json"
    elif sys_key == "b":
        rag = HybridRAG()
        sys_name = "SYSTEM B (HYBRID BM25 + DENSE RRF)"
        report_file = "system_b_hybrid.json"
    else:
        rag = RoutedRAG()
        sys_name = "SYSTEM C (CASCADING INTENT ROUTER)"
        report_file = "system_c_routed.json"

    print("\n" + "=" * 80)
    print(f"       REPOLENS EVALUATION HARNESS — {sys_name}")
    print("=" * 80)
    print(f"Loaded {len(questions)} benchmark queries across 8 evaluation categories.\n")

    results = []

    header = f"{'ID':<6} | {'Category':<16} | {'Recall':<7} | {'Faithful':<8} | {'Refusal':<7} | {'Cite':<5} | {'Latency':<7}"
    print(header)
    print("-" * 80)

    for q in questions:
        query_id = q["id"]
        category = q["category"]
        question_text = q["question"]
        expected_files = q.get("expected_files", [])
        expected_keywords = q.get("expected_keywords", [])
        must_refuse = q.get("must_refuse", False)

        # Run through selected pipeline
        output = rag.answer(question_text)

        # Compute objective and judge metrics
        recall = evaluate_retrieval_recall(output["retrieved_chunks"], expected_files)
        kw_cov = evaluate_keyword_coverage(output["retrieved_chunks"], expected_keywords)
        refusal = evaluate_refusal_accuracy(output["answer"], must_refuse)
        citation = evaluate_citation_presence(output["answer"], output["retrieved_chunks"])
        faithfulness = evaluate_faithfulness(output["answer"], output["retrieved_chunks"])
        latency_sec = output["total_latency_ms"] / 1000.0

        results.append({
            "id": query_id,
            "category": category,
            "question": question_text,
            "recall": recall,
            "keyword_coverage": kw_cov,
            "refusal_accuracy": refusal,
            "citation_presence": citation,
            "faithfulness": faithfulness,
            "latency_sec": latency_sec,
            "answer": output["answer"],
            "retrieved_chunks": output.get("retrieved_chunks", []),
            "routing_decision": output.get("routing_decision"),
        })

        row = f"{query_id:<6} | {category:<16} | {recall*100:>5.1f}% | {faithfulness*100:>7.1f}% | {refusal*100:>6.1f}% | {citation*100:>4.0f}% | {latency_sec:>6.2f}s"
        print(row)

        # Gentle pause between questions to respect free tier quotas
        time.sleep(1.0)

    # Calculate overall averages
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_kw = sum(r["keyword_coverage"] for r in results) / len(results)
    avg_refusal = sum(r["refusal_accuracy"] for r in results) / len(results)
    avg_citation = sum(r["citation_presence"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)

    print("-" * 80)
    print("\n" + "=" * 50)
    print(f"         {sys_name} SCORECARD")
    print("=" * 50)
    print(f"  Retrieval Recall@5:    {avg_recall * 100:.1f}%")
    print(f"  Keyword Coverage:      {avg_kw * 100:.1f}%")
    print(f"  Faithfulness (Judge):  {avg_faithfulness * 100:.1f}%")
    print(f"  Refusal Accuracy:      {avg_refusal * 100:.1f}%")
    print(f"  Citation Presence:     {avg_citation * 100:.1f}%")
    print(f"  Avg Total Latency:     {avg_latency:.2f}s")
    print("=" * 50)

    # Save detailed evaluation run to disk
    report_path = reports_dir / report_file
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "system": sys_name,
                "summary": {
                    "recall": avg_recall,
                    "keyword_coverage": avg_kw,
                    "faithfulness": avg_faithfulness,
                    "refusal_accuracy": avg_refusal,
                    "citation_presence": avg_citation,
                    "avg_latency_sec": avg_latency,
                },
                "details": results,
            },
            f,
            indent=2,
        )

    print(f"\n[+] Full report saved to {report_path}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RepoLens Benchmark Evaluation Harness")
    parser.add_argument(
        "--system",
        choices=["a", "b", "c"],
        default="c",
        help="System architecture to evaluate: 'a' (Baseline), 'b' (Hybrid), 'c' (Routed)",
    )
    args = parser.parse_args()
    run_benchmark(args.system)
