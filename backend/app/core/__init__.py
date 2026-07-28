"""Cross-cutting configuration, logging, and security utilities.

Modules in this package are imported by most backend layers.  They provide
environment settings, the redacting logger, and the process-local secret
store; none of them should contain request-specific or media-pipeline logic.
"""
