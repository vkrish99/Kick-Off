"""
Structured Logging Configuration
=================================
Provides a JSON formatter and a helper to configure the root logger.
Use JSON mode in production for log aggregator compatibility (e.g. Datadog, Loki).
"""
import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Emits each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        obj: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj)


def configure_logging(json_logs: bool = False) -> None:
    """
    Configure the root logger.

    Args:
        json_logs: When True, format records as JSON (recommended for production).
                   When False, use a human-readable format (recommended for development).
    """
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
