"""D17 local-model/API/real-MP4 acceptance using an isolated synthetic database."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


async def run(args: argparse.Namespace) -> dict[str, Any]:
    # Configure isolation before importing DB, API, workers or provider factories.
    from app.core import config
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    storage = output / f"synthetic-storage-{uuid.uuid4().hex[:12]}"
    settings = config.Settings(
        _env_file=None, storage_root=storage,
        database_url=f"sqlite:///{(storage / 'test.db').as_posix()}",
        language_model=args.model, language_base_url=args.base_url,
        language_reasoning_effort="none", language_review_all=False,
        llm_api_key=None, image_api_key=None, output_width=1280, output_height=720, output_fps=15,
    )
    config.get_settings = lambda: settings
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select
    from app.db import get_session_factory, init_db
    from app.main import create_app
    from app.models.job import GenerationJob
    from app.services.pipeline import run_generation_job
    from app.workers.job_runner import job_registry
    init_db()
    records: list[dict[str, Any]] = []
    client = TestClient(create_app())  # No dispatcher lifespan: inspect queue before running it.

    def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = client.post(path, json=body)
        if response.status_code != 200:
            raise AssertionError(f"HTTP {response.status_code}: {response.text[:500]}")
        return response.json()

    def jobs() -> int:
        with get_session_factory()() as db:
            return db.scalar(select(func.count()).select_from(GenerationJob))

    def project(project_id: int) -> dict[str, Any]:
        response = client.get(f"/api/projects/{project_id}")
        assert response.status_code == 200
        return response.json()

    def nl(project_id: int, name: str, text: str) -> dict[str, Any]:
        response = post("/api/language/requests", {
            "request_id": name, "text": text, "target": {"selected_project_id": project_id},
        })
        records.append({"case": name, "synthetic_text": text, "response": response})
        print(json.dumps({"case": name, "status": response["status"],
                          "operation": (response.get("prepared_request") or {}).get("operation_id"),
                          "failure": response.get("failure")}, ensure_ascii=False), flush=True)
        return response

    def typed(project_id: int, name: str, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return post("/api/operations/execute", {"request_id": name, "operation_id": operation,
            "arguments": arguments, "target": {"project_id": project_id},
            "base_revision": project(project_id)["revision"]})

    def same_settings(first: int, second: int) -> None:
        from app.services.settings_history import SETTING_FIELDS
        left, right = project(first), project(second)
        for field in (*SETTING_FIELDS, "revision"):
            assert left[field] == right[field], f"saved settings differ: {field}"

    payload = {"title": "D17 synthetic video", "source_script": "これは自然言語から動画を作る合成テストです。設定の保存と動画の生成を分けて確認します。",
               "subtitle_font_size": 48, "use_fake_providers": True}
    project_ids: list[int] = []
    for _ in range(2):
        response = client.post("/api/projects", json=payload)
        assert response.status_code == 201
        project_ids.append(response.json()["id"])
    first, second = project_ids
    try:
        result = nl(first, "set", "選択中のプロジェクトの字幕を56pxにしてください。")
        assert result["status"] == "completed", result
        typed(second, "typed-set", "project.subtitle-font-size.set", {"value": 56})
        same_settings(first, second)
        assert jobs() == 0

        result = nl(first, "adjust", "字幕を少し大きくして。")
        assert result["status"] == "completed", result
        typed(second, "typed-adjust", "project.subtitle-font-size.adjust", {"delta": 2})
        same_settings(first, second)
        assert project(first)["subtitle_font_size"] == 58 and jobs() == 0

        result = nl(first, "status", "選択中の動画の状態を教えて。")
        expected = typed(second, "typed-status", "project.status.get", {})
        assert result["status"] == "completed", result
        assert result["result"]["data"] == expected["data"]
        same_settings(first, second)

        result = nl(first, "missing-value", "字幕の文字サイズを変更して。")
        assert result["status"] == "needs_input" and not result["executed"], result
        result = nl(first, "unsupported", "このアプリから友達にメールを送って。")
        assert result["status"] == "unsupported" and not result["executed"], result
        same_settings(first, second)
        assert jobs() == 0

        result = nl(first, "no-generation", "字幕を60pxにして。動画の生成はしないで。")
        assert result["status"] == "completed", result
        typed(second, "typed-no-generation", "project.subtitle-font-size.set", {"value": 60})
        same_settings(first, second)
        assert jobs() == 0

        result = nl(first, "generate", "現在の設定で動画を作り直して。")
        assert result["status"] == "ready" and result["requires_confirmation"], result
        assert result["prepared_request"]["operation_id"] == "project.generation.start", result
        assert jobs() == 0
        confirm_url = "/api/language/requests/generate/execute"
        refused = client.post(confirm_url, json={"confirmation_token": result["confirmation_token"]})
        assert refused.status_code == 409 and jobs() == 0
        confirmation = {"confirmation_token": result["confirmation_token"], "confirm_generation": True}
        completed = post(confirm_url, confirmation)
        records.append({"case": "explicit-generation-confirmation", "response": completed})
        expected = typed(second, "typed-generate", "project.generation.start", {"kind": "full"})
        assert post(confirm_url, confirmation) == completed
        assert jobs() == 2
        job_ids = [completed["result"]["job_id"], expected["job_id"]]
        tasks = [job_registry.submit(job_id, lambda cancel, jid=job_id: run_generation_job(jid, cancel)) for job_id in job_ids]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)
        videos: list[dict[str, Any]] = []
        for index, project_id in enumerate(project_ids):
            saved = project(project_id)
            assert saved["status"] == "completed", saved
            history = client.get(f"/api/projects/{project_id}/history").json()
            assert history["output_state"] == "current" and len(history["artifacts"]) == 1, history
            artifact = history["artifacts"][0]
            video = client.get(artifact["video_url"])
            assert video.status_code == 200
            path = output / ("natural-language-demo.mp4" if index == 0 else "typed-api-demo.mp4")
            path.write_bytes(video.content)
            probe = subprocess.run([config.resolve_ffprobe(settings), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                                   capture_output=True, text=True, timeout=30, check=True)
            subprocess.run([config.resolve_ffmpeg(settings), "-v", "error", "-i", str(path), "-f", "null", "-"],
                           capture_output=True, timeout=30, check=True)
            videos.append({"path": str(path), "artifact": artifact, "media": json.loads(probe.stdout),
                           "revision": saved["revision"], "subtitle_font_size": saved["subtitle_font_size"]})
        assert videos[0]["revision"] == videos[1]["revision"] == 4
        assert videos[0]["subtitle_font_size"] == videos[1]["subtitle_font_size"] == 60
        assert videos[0]["media"]["format"]["duration"] == videos[1]["media"]["format"]["duration"]
        return {"passed": True, "created_at": datetime.now(timezone.utc).isoformat(), "mode": "all_tools",
                "model": args.model, "base_url": args.base_url, "reasoning_effort": "none",
                "data": "synthetic only; isolated DB; fake video providers", "external_api_cost_yen": 0,
                "storage": str(storage), "records": records, "videos": videos}
    except BaseException:
        (output / "incomplete-probe.json").write_text(json.dumps({"passed": False, "records": records,
            "storage": str(storage)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    evidence = asyncio.run(run(args))
    (args.output_dir / "all-tools-probe.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("D17 All Tools + typed API equivalence + MP4: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
