# BlockVideo Makefile
# ---------------------------------------------------------------
# Common workflows:
#   make install         - install backend + frontend deps
#   make backend         - run the FastAPI backend (port 8000)
#   make frontend        - run the Vite dev server (port 5173)
#   make test            - run backend + frontend tests
#   make demo            - end-to-end smoke run (FakeProvider) producing a sample MP4
#   make clean           - remove generated storage + build artifacts
# ---------------------------------------------------------------

BACKEND_DIR    := backend
FRONTEND_DIR   := frontend
SAMPLES_DIR    := samples
STORAGE_DIR    := storage
SAMPLE_SCRIPT  := $(SAMPLES_DIR)/compose_multiplatform_intro.txt
API            := http://127.0.0.1:8000

UV             ?= uv
PNPM           ?= pnpm
# The backend venv always has Python; bare `python` is missing on stock Debian/Ubuntu.
PYTHON         ?= $(UV) run --project $(BACKEND_DIR) python
# backend/.env may name ffmpeg/ffprobe by absolute path when they are not on PATH.
FFMPEG_PATH    ?= $(strip $(shell sed -n 's/^FFMPEG_PATH=//p' $(BACKEND_DIR)/.env 2>/dev/null | tr -d '"\r'))
FFMPEG         ?= $(if $(FFMPEG_PATH),$(FFMPEG_PATH),ffmpeg)
FFPROBE_PATH   ?= $(strip $(shell sed -n 's/^FFPROBE_PATH=//p' $(BACKEND_DIR)/.env 2>/dev/null | tr -d '"\r'))
FFPROBE        ?= $(if $(FFPROBE_PATH),$(FFPROBE_PATH),ffprobe)

# Recipes need a POSIX shell (on Windows: GNU make run from Git Bash or MSYS2).
BACKEND_RUN     := cd $(BACKEND_DIR) && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000
BACKEND_TEST    := cd $(BACKEND_DIR) && $(UV) run pytest
FRONTEND_RUN    := cd $(FRONTEND_DIR) && $(PNPM) dev
FRONTEND_TEST   := cd $(FRONTEND_DIR) && $(PNPM) test
FRONTEND_BUILD  := cd $(FRONTEND_DIR) && $(PNPM) build

.PHONY: help install backend frontend test demo clean lint format check ffmpeg-check

help:
	@echo "BlockVideo"
	@echo "  install      install backend + frontend deps"
	@echo "  backend      run FastAPI backend on :8000"
	@echo "  frontend     run Vite dev server on :5173"
	@echo "  test         run backend + frontend tests"
	@echo "  demo         produce a sample MP4 end-to-end with FakeProvider"
	@echo "  ffmpeg-check verify ffmpeg/ffprobe are on PATH"
	@echo "  lint         run ruff + tsc type-check"
	@echo "  format       run ruff --fix"
	@echo "  clean        remove generated storage + frontend dist"

ffmpeg-check:
	@command -v "$(FFMPEG)" >/dev/null 2>&1 || (echo "ffmpeg not found (PATH or FFMPEG_PATH)" && exit 1)
	@command -v "$(FFPROBE)" >/dev/null 2>&1 || (echo "ffprobe not found (PATH or FFPROBE_PATH)" && exit 1)
	@"$(FFMPEG)" -version | head -n 1
	@"$(FFPROBE)" -version | head -n 1

install:
	$(UV) sync --project $(BACKEND_DIR) --extra dev
	cd $(FRONTEND_DIR) && $(PNPM) install

backend:
	$(BACKEND_RUN)

frontend:
	$(FRONTEND_RUN)

test:
	$(BACKEND_TEST)
	$(FRONTEND_TEST)

lint:
	cd $(BACKEND_DIR) && $(UV) run ruff check .
	cd $(FRONTEND_DIR) && $(PNPM) exec tsc -b

format:
	cd $(BACKEND_DIR) && $(UV) run ruff check --fix .

check: lint test

# The driver owns the backend process: it is stopped (with its children, also
# on Windows) even when a step fails, and a health timeout is an error.
demo: ffmpeg-check
	@$(PYTHON) scripts/demo_run.py --serve
	@echo "[demo] done. Output:"
	@find $(STORAGE_DIR) -name '*.mp4' -type f -exec ls -lh {} \;

clean:
	rm -rf $(STORAGE_DIR)
	rm -rf $(BACKEND_DIR)/.pytest_cache $(BACKEND_DIR)/.ruff_cache
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.vite