"""SQLAlchemy ORM models for projects, blocks, and generation jobs.

``project.Project`` is the parent row, ``block.Block`` stores one script
chunk and its artifacts, and ``job.GenerationJob`` records a long-running
pipeline invocation.  ``app.db.init_db`` imports the concrete modules so all
three model families register their tables with ``db.Base``.
"""
