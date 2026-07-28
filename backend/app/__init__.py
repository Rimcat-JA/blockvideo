"""Package metadata shared by the BlockVideo backend.

The package has no application logic of its own.  ``__version__`` is imported
by the FastAPI factory and the health endpoint, while ``APP_NAME`` is a stable
human-readable identifier for diagnostics and future integrations.
"""

# Public API version reported by FastAPI and ``GET /api/health``.
__version__ = "0.1.0"
# Display name used when identifying the backend in logs or tooling.
APP_NAME = "BlockVideo"
