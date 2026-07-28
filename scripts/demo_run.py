"""End-to-end demo runner.

Reads ``samples/compose_multiplatform_intro.txt`` and POSTs it to a locally-running
BlockVideo backend, then polls until the project reaches the ``completed`` state
(or fails) and prints the final artifact paths.

Requires:
    * Backend already running on http://127.0.0.1:8000 (started by the Makefile)
    * Python 3.10+ (no third-party deps — uses urllib only)

Usage:
    python scripts/demo_run.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("BLOCKVIDEO_API", "http://127.0.0.1:8000")
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "samples" / "compose_multiplatform_intro.txt"
POLL_INTERVAL_SEC = 1.0
TIMEOUT_SEC = 180.0


def _request(
    method: str, path: str, payload: dict | None = None
) -> tuple[int, dict | str]:
    url = f"{API_BASE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def _wait_for_status(project_id: int, target: set[str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        status_code, body = _request("GET", f"/api/projects/{project_id}")
        if status_code != 200:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        assert isinstance(body, dict)
        last = body
        current = body.get("status", "?")
        progress = body.get("progress", 0.0)
        stage = body.get("current_stage") or "-"
        print(f"  status={current:>11}  progress={progress:5.1%}  stage={stage}")
        if current in target:
            return body
        if current == "failed":
            raise SystemExit(f"project failed: {body.get('error_message')}")
        time.sleep(POLL_INTERVAL_SEC)
    raise SystemExit(f"timed out after {timeout}s waiting for {target}")


def main() -> int:
    if not SCRIPT_PATH.exists():
        print(f"sample script not found: {SCRIPT_PATH}", file=sys.stderr)
        return 1
    source_script = SCRIPT_PATH.read_text(encoding="utf-8").strip()
    print(f"[demo] using sample script ({len(source_script)} chars): {SCRIPT_PATH.name}")

    print("[demo] POST /api/projects (use_fake_providers=True)")
    status_code, body = _request(
        "POST",
        "/api/projects",
        {
            "title": "デモ: Compose Multiplatform 入門",
            "source_script": source_script,
            "use_fake_providers": True,
        },
    )
    if status_code not in (200, 201):
        print(f"  failed: {status_code} {body}", file=sys.stderr)
        return 1
    assert isinstance(body, dict)
    project_id = body["id"]
    print(f"  -> project_id={project_id}")

    print("[demo] POST /api/projects/{id}/generate-all")
    status_code, body = _request("POST", f"/api/projects/{project_id}/generate-all")
    if status_code not in (200, 202):
        print(f"  failed: {status_code} {body}", file=sys.stderr)
        return 1
    assert isinstance(body, dict)
    job = body.get("job") or {}
    print(f"  -> job_id={job.get('id')} stage={job.get('current_stage')}")

    print("[demo] polling project status...")
    final = _wait_for_status(project_id, target={"completed", "failed"}, timeout=TIMEOUT_SEC)

    print("[demo] listing blocks:")
    status_code, blocks = _request("GET", f"/api/projects/{project_id}/blocks")
    if status_code == 200 and isinstance(blocks, list):
        for b in blocks:
            print(
                f"  block {b['index']}: type={b['visual_type']:>12} "
                f"audio_ms={b['duration_ms']} display_ms={b['display_duration_ms']}"
            )

    print("[demo] fetching artifacts:")
    out_video = final.get("output_video_path")
    out_sub = final.get("output_subtitle_path")
    if out_video:
        print(f"  video  -> {out_video}")
    if out_sub:
        print(f"  ass    -> {out_sub}")
    print(f"  download endpoint -> {API_BASE}/api/projects/{project_id}/download")
    return 0


if __name__ == "__main__":
    sys.exit(main())