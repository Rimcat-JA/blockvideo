"""Isolated loopback demo of the normal application, selected at process startup."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core import config

FORMAT = "blockvideo-plan-c-demo-v1"


def prepare_storage(directory: Path) -> Path:
    directory = directory.resolve()
    marker = directory / "demo.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")).get("format") != FORMAT:
            raise ValueError("unrecognized demo storage")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("use an empty directory or previously marked demo storage")
        directory.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    return directory


@contextmanager
def exclusive_demo(directory: Path) -> Iterator[None]:
    """Keep mode switching sequential; a crashed process automatically releases the lock."""
    with (directory / "server.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def demo_settings(directory: Path, mode: str, model: str, index: Path | None) -> config.Settings:
    if mode not in {"all_tools", "stateful"}:
        raise ValueError("only All Tools and retained-state retrieval are available")
    if mode == "stateful" and (index is None or not index.is_dir()):
        raise ValueError("stateful mode requires an existing index directory")
    return config.Settings(_env_file=None, storage_root=directory,
        database_url=f"sqlite:///{(directory / 'demo.db').as_posix()}",
        language_model=model, language_base_url="http://127.0.0.1:1234/v1", language_reasoning_effort="none",
        language_retrieval_index=index.resolve() if mode == "stateful" else None,
        language_retrieval_profile=config.PROJECT_ROOT / "app/retrieval/e5-profile.json",
        language_embedding_assets=config.PROJECT_ROOT / "storage/embedding-models/multilingual-e5-small",
        language_embedding_base_url="http://127.0.0.1:1234/v1",
        language_retrieval_readiness=True, language_retrieval_all_tools=True, language_review_all=False,
        llm_api_key=None, image_api_key=None, output_width=1280, output_height=720, output_fps=15)


def create_demo_app(settings: config.Settings, frontend: Path) -> Any:
    """Call in a fresh interpreter before importing app modules that capture settings."""
    if not (frontend / "index.html").is_file():
        raise ValueError("build the frontend first")
    config.get_settings = lambda: settings
    from starlette.exceptions import HTTPException
    from starlette.staticfiles import StaticFiles
    from app.main import app

    class SPAFiles(StaticFiles):
        async def get_response(self, path: str, scope: Any) -> Any:
            try:
                return await super().get_response(path, scope)
            except HTTPException as exc:
                if exc.status_code != 404 or path == "api" or path.startswith("api/") or Path(path).suffix:
                    raise
                return await super().get_response("index.html", scope)

    app.mount("/", SPAFiles(directory=frontend, html=True), name="demo-ui")
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--mode", choices=("all_tools", "stateful"))
    parser.add_argument("--model", default="ternary-bonsai-27b-heretic-ja")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    mode = args.mode
    if mode is None:
        print("1: 検索なし（全操作から判断）\n2: 状態付き検索\n終了後、同じ保存先で再起動すると方式を切り替えられます。")
        choice = input("1 または 2: ").strip()
        if choice not in {"1", "2"}:
            parser.error("choose 1 or 2")
        mode = "all_tools" if choice == "1" else "stateful"
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    frontend = config.REPO_ROOT / "frontend/dist"
    try:
        settings = demo_settings(args.storage.resolve(), mode, args.model, args.index)
        if not (frontend / "index.html").is_file():
            raise ValueError("build the frontend first")
        directory = prepare_storage(args.storage)
        with exclusive_demo(directory):
            app = create_demo_app(settings, frontend)
            with (directory / "launches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "mode": mode,
                    "model": args.model, "port": args.port, "index": str(settings.language_retrieval_index)
                    if settings.language_retrieval_index else None}, ensure_ascii=False) + "\n")
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=args.port)
    except (OSError, ValueError) as exc:
        print(f"Demo not started: {type(exc).__name__}. Check storage lock, empty directory, index and frontend build.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
