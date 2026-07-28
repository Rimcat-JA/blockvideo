"""FastAPI route modules and HTTP boundary helpers.

The package separates HTTP concerns from pipeline implementation details:
route modules validate requests and translate ORM objects into schemas, while
``app.api.utils`` contains shared safety checks for serving stored artifacts.
"""
