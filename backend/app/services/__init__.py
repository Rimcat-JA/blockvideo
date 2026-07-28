"""Pipeline stages and media transformation services.

The modules here are intentionally small boundaries: pure layout and hashing
helpers are kept separate from external providers, while ``pipeline`` is the
only module that coordinates persistence and the complete split-to-video flow.
"""
