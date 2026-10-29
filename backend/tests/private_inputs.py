"""Skip helpers for tests whose inputs are not part of the public repository.

The evaluation corpora under ``evaluation/`` and the pinned development commits
are private records (see .gitignore). A fresh clone has neither, so the tests
that read them skip instead of failing during collection.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[2]


def private_corpus(relative: str, *, module: bool = False) -> Path:
    """Return the private evaluation file, or skip when this checkout lacks it."""
    path = REPOSITORY / relative
    if not path.is_file():
        pytest.skip(f"private evaluation input {relative} is not in this checkout",
                    allow_module_level=module)
    return path


def require_commit(commit: str, repository: Path = REPOSITORY) -> None:
    """Skip unless ``commit`` exists in ``repository`` (pinned private history)."""
    probe = subprocess.run(["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True, timeout=30)
    if probe.returncode != 0:
        pytest.skip(f"pinned commit {commit[:12]} is not in this repository's history")
