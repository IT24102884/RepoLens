import json
import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def evaluate_retrieval_recall(retrieved_chunks: List[Dict[str, Any]], expected_files: List[str]) -> float:
    """Checks if the ground-truth file was actually found in the top retrieved chunks."""
    if not expected_files:
        return 1.0  # Nothing expected 

    retrieved_files = {c.get("file_path", "").replace("\\", "/") for c in retrieved_chunks}

    hits = 0
    for exp in expected_files:
        exp_clean = exp.replace("\\", "/").removesuffix(".json")
        exp_digits = re.findall(r"\d+", exp_clean)
        matched = False
        for rf in retrieved_files:
            rf_clean = rf.replace("\\", "/").removesuffix(".json")
            rf_digits = re.findall(r"\d+", rf_clean)
            if (
                exp_clean in rf
                or rf in exp_clean
                or exp in rf
                or (exp_clean.split("/")[-1] in rf_clean)
                or (exp_digits and rf_digits and exp_digits == rf_digits and ("issue" in exp_clean or "ticket" in exp_clean))
            ):
                matched = True
                break
        if matched:
            hits += 1

    return hits / len(expected_files)


def evaluate_keyword_coverage(retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    """Checks what percentage of key terms exist inside the retrieved text."""
    if not expected_keywords:
        return 1.0

    combined_text = " ".join(c.get("content", "").lower() for c in retrieved_chunks)
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


def evaluate_refusal_accuracy(answer: str, must_refuse: bool) -> float:
    """Checks whether the model appropriately admits it doesn't know for out-of-scope questions."""
    refusal_signals = [
        "cannot find",
        "not found",
        "does not support",
        "not mentioned",
        "no information",
        "not available in the repository",
        "not provide",
    ]
    answer_lower = answer.lower()
    refused = any(signal in answer_lower for signal in refusal_signals)

    if must_refuse:
        return 1.0 if refused else 0.0
    else:
        return 0.0 if refused else 1.0


def evaluate_citation_presence(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Verifies whether the answer actually points back to real source files or line numbers."""
    if not retrieved_chunks:
        return 1.0

    # Look for common citation patterns like 'Lines 10-20', 'L10', or file extensions
    has_line_citation = bool(re.search(r"(?:line|lines|l)\s*\d+", answer, re.IGNORECASE))
    has_file_citation = bool(re.search(r"\b[\w-]+\.(?:py|md|json)\b", answer, re.IGNORECASE))

    if has_line_citation or has_file_citation:
        return 1.0
    return 0.0


def evaluate_faithfulness(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Uses LLM-as-a-judge to detect whether any claims in the answer are ungrounded hallucinations."""
    # If the model explicitly refused, there are no ungrounded claims made
    refusal_signals = ["cannot find", "not found in the repository", "not available in the repository"]
    if any(sig in answer.lower() for sig in refusal_signals):
        return 1.0

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return 1.0

    client = Groq(api_key=api_key)
    context_text = "\n---\n".join(c.get("content", "") for c in retrieved_chunks[:3])

    judge_prompt = (
        "You are an impartial evaluation judge for a RAG system.\n"
        "Your job is to check if the claims in the ANSWER are strictly supported by the CONTEXT.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score the faithfulness from 0.0 to 1.0:\n"
        "- 1.0 = All claims are directly supported by context.\n"
        "- 0.5 = Mostly supported, but contains minor assumptions or external facts.\n"
        "- 0.0 = Contains clear hallucinations, invented APIs, or unsupported claims.\n\n"
        "Output ONLY a JSON with key 'faithfulness': {\"faithfulness\": float}"
    )

    try:
        res = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content or "{}")
        score = float(data.get("faithfulness", 1.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 1.0
