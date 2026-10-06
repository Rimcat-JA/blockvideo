"""Catalog-wide and per-model-call operation limits.

The catalog may grow well beyond what one prompt can carry; retrieval narrows it
before any model call. Only a single model call is bounded by the prompt limit.
"""
from __future__ import annotations

# Operation versions the catalog, retrieval scope, ranking and diagnostics may hold.
MAX_CATALOG_OPERATIONS = 4096
# Operation versions offered to the model in one interpretation call.
MAX_PROMPT_CANDIDATES = 32
# Ordered steps in one multi-step plan (each step is an ordinary proposal).
MAX_PLAN_STEPS = 4
# Index documents (descriptions, examples, schema lines, annotations) and the bundle
# size. Vectors are stored as JSON floats, so beyond this a binary vector format is
# needed (16,384 x 768 floats is roughly 0.1-0.2 GB of JSON).
MAX_INDEX_DOCUMENTS = 16384
MAX_INDEX_BUNDLE_BYTES = 256_000_000
