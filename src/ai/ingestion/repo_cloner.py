import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

SUPPORTED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".md",
}

IGNORED_DIRECTORIES = {
    ".git", ".github", "node_modules", "vendor", "dist", "build", "target", "out",
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", "coverage", ".next", ".nuxt", "public", "static",
}

MAX_FILE_SIZE_BYTES = 250_000  # 250 KB cap to avoid minified bundles, lockfiles, data dumps
MAX_FILES_BUDGET = 2_500      # 2,500 core files maximum to ensure fast sub-minute embedding


class RepoCloner:
    """Clones GitHub repositories with shallow depth and applies zero-waste file filtering."""

    @staticmethod
    def parse_repo_slug(repo_url: str) -> Tuple[str, str]:
        """Extract owner and repo name from GitHub URL."""
        cleaned = repo_url.strip().removesuffix(".git").rstrip("/")
        match = re.search(r"(?:^|[/@])github\.com[/:]([\w.-]+)/([\w.-]+)", cleaned)
        if not match:
            raise ValueError(
                f"Invalid GitHub URL: '{repo_url}'. Expected format: 'https://github.com/owner/repo'"
            )
        return match.group(1), match.group(2)

    @classmethod
    def clone(cls, repo_url: str, base_dir: Path = Path("data/repos")) -> Tuple[Path, str]:
        """Execute shallow clone (--depth 1) into managed directory."""
        owner, repo_name = cls.parse_repo_slug(repo_url)
        slug = f"{owner}_{repo_name}"
        repo_dir = base_dir / slug

        if repo_dir.exists():
            return repo_dir, f"{owner}/{repo_name}"

        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "git", "clone",
            "--depth", "1",
            "--single-branch",
            repo_url,
            str(repo_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr.strip() or result.stdout.strip()}")

        return repo_dir, f"{owner}/{repo_name}"

    @classmethod
    def collect_source_files(cls, repo_dir: Path) -> List[Path]:
        """Scan directory and apply zero-waste engineering filters."""
        collected: List[Path] = []

        for root, dirs, files in os.walk(repo_dir):
            # Prune ignored directories in-place so os.walk skips them
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")]

            for file_name in files:
                ext = Path(file_name).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                full_path = Path(root) / file_name

                # Skip files exceeding 250 KB
                try:
                    if full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                        continue
                except OSError:
                    continue

                collected.append(full_path)

        # Sort files by engineering priority: Docs -> Core Src/Lib -> Others -> Tests
        def priority_score(p: Path) -> int:
            rel = p.as_posix().lower()
            if rel.endswith("readme.md") or "/docs/" in rel:
                return 1
            if "/src/" in rel or "/lib/" in rel or "/pkg/" in rel or "/app/" in rel or "/api/" in rel:
                return 2
            if "/test" in rel or "/spec" in rel:
                return 4
            return 3

        collected.sort(key=priority_score)
        return collected[:MAX_FILES_BUDGET]
