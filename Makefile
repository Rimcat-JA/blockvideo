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
PYTHON         ?= python
FFMPEG         ?= ffmpeg

# Detect an OS-specific activation script.
ifeq ($(OS),Windows_NT)
	BACKEND_RUN     := cd $(BACKEND_DIR) && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000
	BACKEND_TEST    := cd $(BACKEND_DIR) && $(UV) run pytest
	FRONTEND_RUN    := cd $(FRONTEND_DIR) && $(PNPM) dev
	FRONTEND_TEST   := cd $(FRONTEND_DIR) && $(PNPM) test
	FRONTEND_BUILD  := cd $(FRONTEND_DIR) && $(PNPM) build
else
	BACKEND_RUN     := cd $(BACKEND_DIR) && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000
	BACKEND_TEST    := cd $(BACKEND_DIR) && $(UV) run pytest
	FRONTEND_RUN    := cd $(FRONTEND_DIR) && $(PNPM) dev
	FRONTEND_TEST   := cd $(FRONTEND_DIR) && $(PNPM) test
	FRONTEND_BUILD  := cd $(FRONTEND_DIR) && $(PNPM) build
endif

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
	@command -v $(FFMPEG) >/dev/null 2>&1 || (echo "ffmpeg not found on PATH" && exit 1)
	@$(FFMPEG) -version | head -n 1

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

demo: ffmpeg-check
	@echo "[demo] starting backend in background..."
	@mkdir -p $(STORAGE_DIR)
	@cd $(BACKEND_DIR) && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000 > ../$(STORAGE_DIR)/demo-backend.log 2>&1 & echo $$! > ../$(STORAGE_DIR)/demo-backend.pid
	@echo "[demo] waiting for /api/health..."
	@for i in $$(seq 1 30); do \
		if curl -sf $(API)/api/health >/dev/null 2>&1; then break; fi; \
		sleep 1; \
	done
	@echo "[demo] creating demo project via /api/projects..."
	@$(PYTHON) scripts/demo_run.py
	@echo "[demo] stopping backend..."
	@kill `cat $(STORAGE_DIR)/demo-backend.pid` 2>/dev/null || true
	@rm -f $(STORAGE_DIR)/demo-backend.pid
	@echo "[demo] done. Output:"
	@find $(STORAGE_DIR) -name '*.mp4' -type f -exec ls -lh {} \;

clean:
	rm -rf $(STORAGE_DIR)
	rm -rf $(BACKEND_DIR)/.pytest_cache $(BACKEND_DIR)/.ruff_cache
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.vite