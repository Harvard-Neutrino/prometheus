import logging
import logging.config
from typing import Optional


def configure_logging(config) -> None:
    """Configure Python logging from a Prometheus config object.

    Reads `config.run.verbosity` (string or int) and optional
    `config.run.logfile` / `config.run.log_format` to set up a console
    handler (and optional file handler) using `logging.config.dictConfig`.
    """
    run_cfg = getattr(config, "run", None)
    level = logging.WARNING
    logfile = None
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if run_cfg is not None:
        verbosity = getattr(run_cfg, "verbosity", None)
        if isinstance(verbosity, str):
            level = getattr(logging, verbosity.upper(), logging.WARNING)
        elif isinstance(verbosity, int):
            level = verbosity
        logfile = getattr(run_cfg, "logfile", None)
        fmt = getattr(run_cfg, "log_format", fmt) or fmt

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": fmt}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            }
        },
        "root": {"handlers": ["console"], "level": level},
    }

    if logfile:
        config_dict["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": level,
            "filename": logfile,
        }
        config_dict["root"]["handlers"].append("file")

    logging.config.dictConfig(config_dict)

    # In user-focused summary mode, silence noisy third-party loggers
    try:
        if run_cfg is not None and getattr(run_cfg, "summary_mode", "user") == "user":
            noisy_loggers = ["jax", "jaxlib", "jax._src.xla_bridge", "absl", "importlib._bootstrap"]
            for name in noisy_loggers:
                try:
                    logging.getLogger(name).setLevel(logging.ERROR)
                except Exception:
                    pass
    except Exception:
        pass
