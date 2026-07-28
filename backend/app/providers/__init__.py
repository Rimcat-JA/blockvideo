"""External-service interfaces and deterministic test providers.

The package separates vendor-neutral protocols (LLM, image, VOICEVOX) from
real HTTP implementations and fake offline implementations.  Service code
should import interfaces/factories rather than constructing vendor clients
directly.
"""
