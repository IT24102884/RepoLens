from pathlib import Path
import pytest
from ai.ingestion.repo_cloner import MAX_FILE_SIZE_BYTES, RepoCloner


def test_parse_repo_slug_variants():
    test_cases = [
        ("https://github.com/encode/httpx", ("encode", "httpx")),
        ("https://github.com/encode/httpx.git", ("encode", "httpx")),
        ("https://github.com/encode/httpx/", ("encode", "httpx")),
        ("git@github.com:tiangolo/fastapi.git", ("tiangolo", "fastapi")),
        ("https://github.com/rust-lang/cargo", ("rust-lang", "cargo")),
    ]
    for url, (expected_owner, expected_repo) in test_cases:
        owner, repo = RepoCloner.parse_repo_slug(url)
        assert owner == expected_owner
        assert repo == expected_repo


def test_parse_repo_slug_invalid():
    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        RepoCloner.parse_repo_slug("https://notgithub.com/foo/bar")

    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        RepoCloner.parse_repo_slug("random_string")


def test_collect_source_files_filtering(tmp_path: Path):
    # Supported files
    (tmp_path / "src").mkdir()
    core_file = tmp_path / "src" / "core.py"
    core_file.write_text("print('hello')", encoding="utf-8")

    doc_file = tmp_path / "README.md"
    doc_file.write_text("# Doc", encoding="utf-8")

    # Ignored directory
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "pkg.js").write_text("console.log()", encoding="utf-8")

    # Ignored extension
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Oversized file
    big_file = tmp_path / "src" / "big.ts"
    big_file.write_text("x" * (MAX_FILE_SIZE_BYTES + 100), encoding="utf-8")

    collected = RepoCloner.collect_source_files(tmp_path)
    collected_names = [p.name for p in collected]

    assert "README.md" in collected_names
    assert "core.py" in collected_names
    assert "pkg.js" not in collected_names
    assert "image.png" not in collected_names
    assert "big.ts" not in collected_names
