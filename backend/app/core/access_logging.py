"""Keep client request identifiers/query strings out of Uvicorn access records."""
from __future__ import annotations

import logging
import re


class RequestPathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.args, tuple) or len(record.args) != 5:
            # Unknown access format must not accidentally emit the original URL.
            return False
        client, method, path, version, status = record.args
        if not isinstance(path, str):
            return False
        path = path.split("?", 1)[0]
        path = re.sub(r"^(/api/(?:language|operations)/requests)/[^/]+", r"\1/:request", path)
        record.args = (client, method, path, version, status)
        return True


def protect_access_logs() -> None:
    target = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, RequestPathFilter) for item in target.filters):
        target.addFilter(RequestPathFilter())
