import time
import logging
import contextlib
from typing import Optional


@contextlib.contextmanager
def time_block(name: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """Context manager that logs the elapsed time for a block.

    Usage:
        with time_block("injection", logger):
            do_work()
    """
    if logger is None:
        logger = logging.getLogger("prometheus")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(level, "%s took %.3f s", name, elapsed)
