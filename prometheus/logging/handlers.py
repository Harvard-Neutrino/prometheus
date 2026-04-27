import logging


class LogCounterHandler(logging.Handler):
    """Simple handler that counts log records by severity.

    The behavior mirrors the previous inline implementation used in
    `prometheus/prometheus.py` and is intentionally minimal: it should
    never raise while counting.
    """

    def __init__(self):
        super().__init__()
        self.warning_count = 0
        self.error_count = 0
        self.critical_count = 0
        self.info_count = 0

    def emit(self, record):
        try:
            if record.levelno >= logging.CRITICAL:
                self.critical_count += 1
            if record.levelno >= logging.ERROR:
                self.error_count += 1
            if record.levelno >= logging.WARNING:
                self.warning_count += 1
            if record.levelno >= logging.INFO:
                self.info_count += 1
        except Exception:
            # Never let the counter interfere with the main logging
            pass


__all__ = ["LogCounterHandler"]
