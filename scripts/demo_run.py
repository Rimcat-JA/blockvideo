"""End-to-end local demo runner for the BlockVideo HTTP API.

Reads ``samples/compose_multiplatform_intro.txt`` and POSTs it to a locally-running
BlockVideo backend, then polls until the project reaches the ``completed`` state
(or fails) and prints the final artifact paths.

Requires:
    * Backend already running on http://127.0.0.1:8000 (started by the Makefile)
    * Python 3.10+ (no third-party deps — uses urllib only)

Usage:
    python scripts/demo_run.py

Imports:
    ``json`` encodes request bodies and decodes API responses.
    ``os`` reads the optional backend base URL.
    ``sys`` reports errors and returns process exit codes.
    ``time`` implements monotonic polling deadlines.
    ``urllib`` performs dependency-free HTTP requests.
    ``Path`` locates the bundled sample script.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Backend origin; override with BLOCKVIDEO_API for a non-default local server.
API_BASE = os.environ.get("BLOCKVIDEO_API", "http://127.0.0.1:8000")
# Sample script submitted by the offline fake-provider demo.
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "samples" / "compose_multiplatform_intro.txt"
# Polling cadence and overall demo deadline.
POLL_INTERVAL_SEC = 1.0
TIMEOUT_SEC = 180.0


def _request(
    method: str, path: str, payload: dict | None = None
) -> tuple[int, dict | str]:
    """Send one HTTP request and decode JSON or text response content.

    Args:
        method: HTTP method such as ``GET`` or ``POST``.
        path: API path appended to ``API_BASE``.
        payload: Optional JSON-serializable request mapping.

    Returns:
        ``(status_code, decoded_body)`` where the body is a mapping when JSON
        decoding succeeds and a string otherwise.  HTTP errors are returned in
        the same tuple shape instead of being raised.

    Raises:
        urllib.error.URLError: For network/DNS failures that are not HTTP
            responses.

    """
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
    """Poll a project until a target status, failure, or timeout.

    Args:
        project_id: Project primary key returned by creation.
        target: Terminal statuses that should return the last project body.
        timeout: Maximum monotonic seconds to wait.

    Returns:
        Last decoded project mapping when one of ``target`` is reached.

    Raises:
        SystemExit: If the project reports ``failed`` or the deadline expires.

    """
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
    """Run the fake-provider sample and print generated artifact paths.

    Returns:
        ``0`` on a completed demo, ``1`` for a missing sample or HTTP failure.

    Side Effects:
        Reads the sample file, creates a project through the backend, queues
        generation, polls status, and prints block/output information.

    """
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
