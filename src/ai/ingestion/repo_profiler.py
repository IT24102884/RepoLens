from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RepoProfile(BaseModel):
    """Dynamic profile describing a software repository's architecture and domain."""
    repo_name: str = Field(description="Repository slug e.g. 'encode/httpx' or 'owner/project'")
    description: str = Field(description="Summary of project domain and purpose")
    primary_languages: List[str] = Field(default_factory=list, description="Detected programming languages")
    subsystems: List[str] = Field(default_factory=list, description="Detected subsystems e.g. ['frontend', 'backend', 'docs']")
    dependencies: List[str] = Field(default_factory=list, description="Key manifest dependencies")
    total_files: int = Field(default=0, description="Total eligible source files indexed")


DEFAULT_HTTPX_PROFILE = RepoProfile(
    repo_name="encode/httpx",
    description="A next-generation HTTP client for Python, supporting HTTP/1.1 and HTTP/2, async and sync APIs.",
    primary_languages=["python"],
    subsystems=["core_client", "transports", "docs", "issues"],
    dependencies=["httpcore", "certifi", "sniffio", "idna", "anyio"],
    total_files=1636,
)


class RepoProfiler:
    """Auto-profiles any repository by inspecting root manifests, dependency files, and README."""

    @staticmethod
    def extract_readme_summary(repo_dir: Path) -> str:
        """Extract high-level description from README.md if present."""
        for name in ["README.md", "readme.md", "README.rst", "README"]:
            readme_path = repo_dir / name
            if readme_path.exists():
                try:
                    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline() for _ in range(35)]

                    cleaned = []
                    for line in lines:
                        line_str = line.strip()
                        # Skip markdown headers, badge links, and empty lines
                        if not line_str or line_str.startswith("#") or line_str.startswith("[!") or line_str.startswith("<img"):
                            continue
                        cleaned.append(line_str)

                    if cleaned:
                        return " ".join(cleaned[:3])[:300]
                except Exception:
                    pass
        return "Software repository codebase."

    @classmethod
    def profile(cls, repo_dir: Path, repo_name: Optional[str] = None) -> RepoProfile:
        repo_dir = Path(repo_dir)
        inferred_name = repo_name or repo_dir.name
        description = cls.extract_readme_summary(repo_dir)

        primary_languages = set()
        subsystems = set()
        dependencies = []

        # 1. Inspect package.json (Node/TypeScript/React)
        pkg_json = repo_dir / "package.json"
        if not pkg_json.exists():
            for sub in repo_dir.glob("*/package.json"):
                pkg_json = sub
                break

        if pkg_json.exists():
            primary_languages.add("typescript" if list(repo_dir.glob("**/*.ts")) else "javascript")
            subsystems.add("frontend")
            try:
                with open(pkg_json, "r", encoding="utf-8", errors="ignore") as f:
                    pkg_data = json.load(f)
                    if not description or description == "Software repository codebase.":
                        description = pkg_data.get("description", description)
                    deps = list(pkg_data.get("dependencies", {}).keys())
                    dependencies.extend(deps[:15])
            except Exception:
                pass

        # 2. Inspect go.mod (Go)
        go_mod = repo_dir / "go.mod"
        if not go_mod.exists():
            for sub in repo_dir.glob("*/go.mod"):
                go_mod = sub
                break

        if go_mod.exists():
            primary_languages.add("go")
            subsystems.add("backend")
            try:
                with open(go_mod, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip().startswith("require") or "\t" in line:
                            parts = line.strip().split()
                            if parts and "/" in parts[0]:
                                dependencies.append(parts[0])
            except Exception:
                pass

        # 3. Inspect pyproject.toml / requirements.txt (Python)
        pyproject = repo_dir / "pyproject.toml"
        req_txt = repo_dir / "requirements.txt"
        if pyproject.exists() or req_txt.exists() or list(repo_dir.glob("**/*.py")):
            primary_languages.add("python")
            if not subsystems:
                subsystems.add("backend")
            if pyproject.exists():
                try:
                    with open(pyproject, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
                        if desc_match and (not description or description == "Software repository codebase."):
                            description = desc_match.group(1)
                except Exception:
                    pass

        # 4. Inspect Cargo.toml (Rust)
        cargo_toml = repo_dir / "Cargo.toml"
        if cargo_toml.exists() or list(repo_dir.glob("**/*.rs")):
            primary_languages.add("rust")
            subsystems.add("backend")
            if cargo_toml.exists():
                try:
                    with open(cargo_toml, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
                        if desc_match and (not description or description == "Software repository codebase."):
                            description = desc_match.group(1)
                except Exception:
                    pass

        # 5. Check for Docs / Issue trackers
        if (repo_dir / "docs").exists() or list(repo_dir.glob("**/*.md")):
            subsystems.add("docs")
        if (repo_dir / "issues").exists():
            subsystems.add("issues")

        return RepoProfile(
            repo_name=inferred_name,
            description=description,
            primary_languages=sorted(list(primary_languages)),
            subsystems=sorted(list(subsystems)),
            dependencies=dependencies[:15],
            total_files=0,
        )
