from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

PINNED_D35 = "522775516c0797abdb313e3432339a3a444b7ae2"

_INDEX_PROBE = r'''
import hashlib, json, sys
from pathlib import Path
from app.retrieval import builder, reader, sources, contracts
from app.retrieval.serialization import RetrievalError
root, index, profile_path, receipt = map(Path, sys.argv[1:])
profile = contracts.EmbeddingProfile.model_validate_json(profile_path.read_bytes())
source = sources.load_sources()
vectors = tuple((1.0, 0.0) if d.key == ("project.subtitle-font-size.set", 1)
                else (0.0, 1.0) for d in source.documents)
manifest = builder.publish_index(index, source, profile, vectors)
verified = reader.load_index(index, source, profile)
assert verified.bundle.documents == source.documents
assert all(Path(m.__file__).resolve().is_relative_to(root / "backend")
           for m in (builder, reader, sources, contracts))
catalog = root / "backend/app/operations/definitions.json"
scope = root / "backend/app/operations/search_scope.json"
negative_root = receipt.parent / "negative-inputs"
negative_root.mkdir()
errors = {}
for name, changed_catalog, changed_scope, changed_profile, changed_index in (
    ("stale_catalog", True, False, False, False),
    ("stale_scope", False, True, False, False),
    ("stale_profile", False, False, True, False),
    ("missing_bundle", False, False, False, True),
):
    c, s, p, i = catalog, scope, profile, index
    if changed_catalog:
        c = negative_root / "catalog.json"
        c.write_bytes(catalog.read_bytes() + b" ")
    if changed_scope:
        s = negative_root / "scope.json"
        s.write_bytes(scope.read_bytes() + b" ")
    if changed_profile:
        p = profile.model_copy(update={"weights_sha256": "1" * 64})
    if changed_index:
        i = negative_root / "missing-index"
        i.mkdir()
        (i / "manifest.json").write_bytes((index / "manifest.json").read_bytes())
    try:
        reader.load_index(i, sources.load_sources(c, s), p)
    except RetrievalError as failure:
        errors[name] = failure.code
    else:
        raise AssertionError("invalid index accepted")
result = {"candidate_commit": "522775516c0797abdb313e3432339a3a444b7ae2",
          "catalog_sha256": manifest.catalog_sha256,
          "scope_sha256": manifest.scope_sha256,
          "bundle_sha256": manifest.bundle_sha256,
          "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
          "document_count": len(verified.bundle.documents),
          "operation_count": manifest.operation_count, "negative_reasons": errors,
          "candidate_module_origins_verified": True}
receipt.write_text(json.dumps(result, sort_keys=True), encoding="ascii")
'''


def source_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def verified_archive(repository: Path, root: Path) -> dict[str, str]:
    tree = subprocess.check_output(["git", "-C", str(repository), "ls-tree", "-r", "-z", PINNED_D35])
    expected: dict[str, str] = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, digest = metadata.split()
        assert mode in (b"100644", b"100755") and kind == b"blob"
        expected[name.decode("utf-8")] = digest.decode("ascii")
    archive = root.parent / "pinned-d35.tar"
    with archive.open("xb") as output:
        subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-C", str(repository), "archive", "--format=tar", PINNED_D35],
                       stdout=output, stderr=subprocess.PIPE, check=True, timeout=30)
    root.mkdir()
    with tarfile.open(archive) as entries:
        for member in entries:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in member.name
            target = root.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            assert member.isfile() and member.name in expected and member.size <= 8 * 1024 * 1024
            stream = entries.extractfile(member)
            assert stream is not None
            with stream:
                raw = stream.read(8 * 1024 * 1024 + 1)
            assert len(raw) == member.size
            object_id = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
            assert object_id == expected[member.name]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(raw)
    hashes = source_hashes(root)
    assert set(hashes) == set(expected)
    archive.unlink()
    return hashes


def build_verified_index(candidate: Path, index: Path, profile: Path, receipt: Path) -> dict[str, Any]:
    env = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
           if name in os.environ}
    env.update({"PYTHONPATH": str(candidate / "backend"), "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _INDEX_PROBE, str(candidate), str(index), str(profile), str(receipt)],
        cwd=candidate / "backend", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")[:4096]
    return json.loads(receipt.read_bytes())
