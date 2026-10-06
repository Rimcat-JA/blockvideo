"""The catalog compiler refuses any disagreement between catalog-side files."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.operations.catalog_compiler import OPERATIONS_DIR, PROMPT_RULES, compile_catalog

FILES = ("definitions.json", "operation_policies.json", "operation_annotations.json", "search_scope.json")


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    for name in FILES:
        shutil.copy(OPERATIONS_DIR / name, tmp_path / name)
    shutil.copy(PROMPT_RULES, tmp_path / "prompt_rules.json")
    return tmp_path


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_repository_catalog_is_consistent() -> None:
    report = compile_catalog()
    assert report.ok, report.errors
    assert report.operations == 10 and report.annotated_operations == 10 and report.prompt_rules > 100


def test_operation_without_policy_is_refused(copy: Path) -> None:
    _edit(copy / "operation_policies.json", lambda data: data["operations"].pop("project.generation.cancel"))
    report = compile_catalog(copy, copy / "prompt_rules.json")
    assert any("without a policy" in error for error in report.errors)


def test_unknown_names_in_annotations_rules_or_scope_are_refused(copy: Path) -> None:
    _edit(copy / "operation_annotations.json",
          lambda data: data["operations"].update({"project.typo.op": {"utterances": ["x"]}}))
    _edit(copy / "prompt_rules.json",
          lambda data: data["rules"].append({"operations": ["project.typo.op"], "text": "x"}))
    _edit(copy / "search_scope.json", lambda data: data["bindings"].pop())
    report = compile_catalog(copy, copy / "prompt_rules.json")
    joined = " | ".join(report.errors)
    assert "annotations" in joined and "prompt rule" in joined and "search scope differs" in joined


def test_cli_exit_code_reflects_the_report(copy: Path) -> None:
    backend = Path(__file__).parents[1]
    ok = subprocess.run([sys.executable, "-B", "-m", "scripts.check_catalog"], cwd=backend, capture_output=True)
    assert ok.returncode == 0, ok.stderr
    _edit(copy / "operation_policies.json", lambda data: data["operations"].pop("project.status.get"))
    bad = subprocess.run([sys.executable, "-B", "-m", "scripts.check_catalog", "--catalog-dir", str(copy),
                          "--prompt-rules", str(copy / "prompt_rules.json")], cwd=backend, capture_output=True)
    assert bad.returncode == 1


def test_prompt_rules_with_unknown_modes_or_features_are_refused(copy: Path) -> None:
    _edit(copy / "prompt_rules.json", lambda data: data["rules"].extend([
        {"modes": ["fast"], "text": "x"}, {"feature": "recipes", "text": "y"}]))
    report = compile_catalog(copy, copy / "prompt_rules.json")
    joined = " | ".join(report.errors)
    assert "unknown mode" in joined and "unknown feature" in joined
