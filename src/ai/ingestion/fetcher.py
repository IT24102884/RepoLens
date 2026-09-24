import io
import json
import zipfile
from pathlib import Path
import httpx

HTTPX_ZIP_URL = "https://github.com/encode/httpx/archive/refs/tags/0.27.0.zip"

SAMPLE_ISSUES = [
    {
        "number": 1240,
        "title": "AsyncClient hangs indefinitely on streaming response with unclosed stream",
        "author": "dev_alex",
        "state": "closed",
        "labels": ["bug", "async", "streaming"],
        "body": "When calling client.stream('GET', url) without using an async with context manager or explicitly closing the response, connection pool exhausts and subsequent requests hang.",
        "comments": [
            {"author": "florimondmanca", "body": "Streaming responses must always be used as context managers or explicitly read to release the underlying connection back to the pool."},
            {"author": "tomchristie", "body": "Fixed in PR #1245 by adding explicit pool timeout warnings and doc guidelines."},
        ],
    },
    {
        "number": 1405,
        "title": "HTTP/2 connection drop on keepalive timeout",
        "author": "cloud_native_sam",
        "state": "closed",
        "labels": ["http2", "network"],
        "body": "Under HTTP/2, long idle connections drop unexpectedly without reconnecting.",
        "comments": [
            {"author": "tomchristie", "body": "We now support httpcore.AsyncConnectionPool(keepalive_expiry=5.0) to safely recycle idle HTTP/2 connections."},
        ],
    },
]


def fetch_httpx_repo(target_dir: Path) -> Path:
    """Download and extract a clean snapshot of encode/httpx into target_dir."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extract_root = target_dir / "httpx-0.27.0"

    if extract_root.exists():
        print(f"[*] Repository already exists at {extract_root}")
        return extract_root

    print("[*] Downloading encode/httpx (tag 0.27.0) snapshot from GitHub...")
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(HTTPX_ZIP_URL)
        response.raise_for_status()

    print("[*] Extracting repository files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(target_dir)

    print(f"[+] Download and extraction complete: {extract_root}")
    return extract_root


def create_issue_fixtures(target_dir: Path) -> Path:
    """Write sample GitHub issue fixtures into data/raw/issues."""
    issues_dir = Path(target_dir) / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)

    for issue in SAMPLE_ISSUES:
        issue_path = issues_dir / f"issue_{issue['number']}.json"
        with open(issue_path, "w", encoding="utf-8") as f:
            json.dump(issue, f, indent=2)

    print(f"[+] Created {len(SAMPLE_ISSUES)} GitHub issue fixtures in {issues_dir}")
    return issues_dir


def main():
    raw_dir = Path("data/raw")
    fetch_httpx_repo(raw_dir)
    create_issue_fixtures(raw_dir)
    print("\n[+] Raw dataset is fully ready for ingestion!")

    # Now run the ingestion pipeline
if __name__ == "__main__":
    main()