import json
from pathlib import Path
from src.core.models import DocumentType
from src.ingestion.pipeline import IngestionPipeline


def test_ingestion_pipeline_end_to_end(tmp_path: Path):
    # 1. Create a mock repository structure in a temporary directory
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    # Add a Python file
    py_file = repo_dir / "client.py"
    py_file.write_text("def fetch_data():\n    return 200\n", encoding="utf-8")

    # Add a Markdown documentation file
    doc_file = repo_dir / "README.md"
    doc_file.write_text("# Project Guide\n\nHow to use this project.\n", encoding="utf-8")

    # Add a GitHub Issue JSON file
    issue_dir = repo_dir / "issues"
    issue_dir.mkdir()
    issue_file = issue_dir / "issue_1.json"
    issue_data = {
        "number": 1,
        "title": "Bug in fetch_data",
        "author": "tester",
        "state": "closed",
        "body": "fetch_data returns int instead of response object.",
    }
    issue_file.write_text(json.dumps(issue_data), encoding="utf-8")

    # Add an ignored file
    ignored_file = repo_dir / "logo.png"
    ignored_file.write_text("fake binary image content", encoding="utf-8")

    # 2. Run the Ingestion Pipeline
    pipeline = IngestionPipeline()
    chunks = pipeline.process_directory(repo_dir)

    # 3. Assertions
    # We should have at least 1 chunk for each source type, and 0 for logo.png
    summary = pipeline.get_summary(chunks)
    assert summary["code_chunks"] >= 1
    assert summary["doc_chunks"] >= 1
    assert summary["ticket_chunks"] >= 1
    assert summary["total_chunks"] == summary["code_chunks"] + summary["doc_chunks"] + summary["ticket_chunks"]

    # Verify that logo.png was completely ignored
    assert not any(c.file_path.endswith(".png") for c in chunks)

    # 4. Verify Saving to JSONL
    output_jsonl = tmp_path / "output" / "chunks.jsonl"
    pipeline.save_chunks(chunks, output_jsonl)
    assert output_jsonl.exists()

    # Check that each line in JSONL is a valid JSON chunk
    with open(output_jsonl, "r", encoding="utf-8") as f:
        saved_lines = [json.loads(line) for line in f]
    assert len(saved_lines) == len(chunks)