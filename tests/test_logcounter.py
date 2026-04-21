import logging

from prometheus.logging.handlers import LogCounterHandler


def test_log_counter_handler_counts():
    h = LogCounterHandler()
    logger = logging.getLogger("prom_test_logger")
    logger.setLevel(logging.DEBUG)
    # Ensure no duplicate handlers from prior runs
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(h)

    logger.info("info")
    logger.warning("warn")
    logger.error("err")
    logger.critical("crit")

    # Basic sanity checks: each counter should be >= 1
    assert h.info_count >= 1
    assert h.warning_count >= 1
    assert h.error_count >= 1
    assert h.critical_count >= 1

    logger.removeHandler(h)
