import json
from pathlib import Path
import pytest
from ai.ingestion.repo_profiler import RepoProfile, RepoProfiler


def test_extract_readme_summary(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# My Awesome Project\n"
        "[![CI](badge.svg)](link)\n\n"
        "A high-throughput distributed database built in Go.\n"
        "Supports linearizable read transactions and Raft consensus.\n",
        encoding="utf-8",
    )
    summary = RepoProfiler.extract_readme_summary(tmp_path)
    assert "A high-throughput distributed database built in Go." in summary
    assert "Supports linearizable read transactions" in summary


def test_profile_python_repo(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "superfast"\nversion = "0.1.0"\n'
        'description = "Ultra fast Python web server"\n'
        'dependencies = ["uvicorn", "pydantic"]\n',
        encoding="utf-8",
    )
    src_file = tmp_path / "main.py"
    src_file.write_text("def run(): pass\n", encoding="utf-8")

    profile = RepoProfiler.profile(tmp_path, repo_name="org/superfast")
    assert profile.repo_name == "org/superfast"
    assert "python" in profile.primary_languages
    assert "backend" in profile.subsystems
    assert "Ultra fast Python web server" in profile.description


def test_profile_node_repo(tmp_path: Path):
    pkg_json = tmp_path / "package.json"
    pkg_data = {
        "name": "react-flow",
        "description": "Interactive node-based UI library",
        "dependencies": {
            "react": "^18.0.0",
            "zustand": "^4.0.0",
        },
    }
    pkg_json.write_text(json.dumps(pkg_data), encoding="utf-8")
    ts_file = tmp_path / "index.ts"
    ts_file.write_text("export const version = '1.0';", encoding="utf-8")

    profile = RepoProfiler.profile(tmp_path, repo_name="org/react-flow")
    assert profile.repo_name == "org/react-flow"
    assert "typescript" in profile.primary_languages
    assert "frontend" in profile.subsystems
    assert "react" in profile.dependencies
    assert "zustand" in profile.dependencies


def test_profile_go_repo(tmp_path: Path):
    go_mod = tmp_path / "go.mod"
    go_mod.write_text(
        "module github.com/org/goservice\n\n"
        "go 1.22\n\n"
        "require (\n"
        "\tgithub.com/gin-gonic/gin v1.9.1\n"
        "\tgoogle.golang.org/grpc v1.62.0\n"
        ")\n",
        encoding="utf-8",
    )
    profile = RepoProfiler.profile(tmp_path, repo_name="org/goservice")
    assert "go" in profile.primary_languages
    assert "backend" in profile.subsystems
    assert "github.com/gin-gonic/gin" in profile.dependencies
