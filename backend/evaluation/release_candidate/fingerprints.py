"""Explicit D35 candidate source allowlist and canonical fingerprints."""
from __future__ import annotations

import subprocess
from pathlib import Path

from evaluation.tool_attestation import (
    FileFingerprint,
    aggregate_fingerprints,
    fingerprint_committed_file,
    validate_git_repository,
)

_REQUIRED_FILES = frozenset(
    {
        "backend/.env.example",
        "backend/app/core/config.py",
        "backend/app/migrations/schema.py",
        "backend/app/operations/definitions.json",
        "backend/app/operations/search_scope.json",
        "backend/app/retrieval/e5-profile.json",
        "backend/app/retrieval/nomic-profile.json",
        "backend/pyproject.toml",
        "backend/uv.lock",
        "docs/DTD.md",
        "docs/plan-c/work-unit-36.md",
        "frontend/package.json",
        "frontend/pnpm-lock.yaml",
        "specification.md",
    }
)
_ROOT_FILES = frozenset(
    {
        ".gitignore",
        "AGENTS.md",
        "LICENSE",
        "Makefile",
        "README.md",
        "development_guideline.md",
        "docker-compose.yml",
        "specification.md",
    }
)
_BACKEND_FILES = frozenset(
    {
        "backend/.dockerignore",
        "backend/.env.example",
        "backend/Dockerfile",
        "backend/pyproject.toml",
        "backend/uv.lock",
    }
)
_FRONTEND_FILES = frozenset(
    {
        "frontend/.pnpm-allow.json",
        "frontend/eslint.config.js",
        "frontend/index.html",
        "frontend/package.json",
        "frontend/pnpm-lock.yaml",
        "frontend/pnpm-workspace.yaml",
        "frontend/postcss.config.js",
        "frontend/tailwind.config.js",
        "frontend/tsconfig.json",
        "frontend/tsconfig.node.json",
        "frontend/vite.config.d.ts",
        "frontend/vite.config.js",
        "frontend/vite.config.ts",
    }
)
_D36_AND_LATER_PREFIXES = (
    "backend/evaluation/release_candidate/",
    "backend/evaluation/scripts/",
)
_D36_AND_LATER_FILES = frozenset(
    {
        "backend/evaluation/final_protocol.json",
        "backend/evaluation/tool_attestation.py",
        "backend/evaluation/unlabeled_contracts.py",
        "backend/tests/test_d36_candidate_protocol.py",
        "backend/tests/test_d36_freeze.py",
        "backend/tests/test_d37_blinded_runner.py",
        "backend/tests/test_d38_result_import.py",
        "backend/tests/test_d39_release_verification.py",
        "backend/tests/test_d40_readiness_decision.py",
    }
)
_EXCLUDED_PARTS = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "dist",
        "node_modules",
        "release-evidence",
        "storage",
    }
)
_EXCLUDED_SUFFIXES = (
    ".ass",
    ".db",
    ".mp3",
    ".mp4",
    ".onnx",
    ".pt",
    ".pth",
    ".sqlite",
    ".sqlite3",
    ".wav",
    ".weights",
)


def _git_tracked_paths(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--cached"],
        check=True,
        capture_output=True,
    )
    paths = completed.stdout.decode("utf-8").split("\0")
    return sorted(path for path in paths if path)


def _is_allowlisted(path: str) -> bool:
    parts = tuple(path.split("/"))
    lowered = path.lower()
    if path in _D36_AND_LATER_FILES or path.startswith(_D36_AND_LATER_PREFIXES):
        return False
    if "held-out" in lowered or "held_out" in lowered:
        return False
    if any(part in _EXCLUDED_PARTS for part in parts):
        return False
    if path in _ROOT_FILES or path in _BACKEND_FILES or path in _FRONTEND_FILES:
        return True
    if parts[-1].startswith(".env") or lowered.endswith(_EXCLUDED_SUFFIXES):
        return False
    if path.startswith("backend/app/"):
        return True
    if path.startswith("backend/evaluation/"):
        return True
    if path.startswith("backend/scripts/") or path.startswith("backend/tests/"):
        return True
    if path.startswith("frontend/src/"):
        return True
    if path.startswith("docs/modules/") and path.endswith(".md"):
        return True
    if path.startswith("docs/implementation-plan-") and path.endswith(".md"):
        return True
    if path.startswith("docs/plan-c/work-unit-") and path.endswith(".md"):
        return True
    if path in {"docs/DTD.md", "docs/plan-c/contracts.md", "docs/plan-c/decisions.md"}:
        return True
    # README-referenced documentation (reports, guides and their images) is part of
    # the candidate, so its README links resolve inside the materialized runtime.
    if path.startswith("docs/") and lowered.endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".svg")):
        return True
    if path.startswith("evaluation/d24/") or path.startswith("evaluation/d31/"):
        return True
    if path.startswith("samples/") or path.startswith("scripts/"):
        return True
    return False


def fingerprint_files(repo_root: Path) -> list[FileFingerprint]:
    commit = validate_git_repository(repo_root)
    tracked = _git_tracked_paths(repo_root)
    missing = sorted(_REQUIRED_FILES.difference(tracked))
    if missing:
        raise ValueError(f"required candidate file is missing: {missing[0]}")
    paths = [path for path in tracked if _is_allowlisted(path)]
    if not paths:
        raise ValueError("candidate fingerprint allowlist is empty")
    files = [fingerprint_committed_file(repo_root, commit, path) for path in paths]
    validate_git_repository(repo_root, expected_commit=commit)
    return files


__all__ = ["aggregate_fingerprints", "fingerprint_files"]
