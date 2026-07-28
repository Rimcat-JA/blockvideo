"""Process-local background job execution helpers.

The current worker implementation uses ``asyncio`` tasks in the API process.
The public enqueue functions hide that implementation so a later queue
backend can replace ``job_runner`` without changing route handlers.
"""
