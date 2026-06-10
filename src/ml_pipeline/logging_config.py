from __future__ import annotations

import logging
import os
import sys

# Third-party loggers that flood output at DEBUG without adding pipeline value.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "google",
    "urllib3",
    "matplotlib",
    "matplotlib.font_manager",
    "matplotlib.pyplot",
    "PIL",
    "fontTools",
    "sklearn",
)


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging for pipeline CLI entry points."""
    env_level = os.getenv("LOG_LEVEL")
    effective = env_level if env_level else level
    numeric = logging.getLevelName(effective) if isinstance(effective, str) else effective
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s %(name)s - %(message)s")
        )
        root.addHandler(handler)
    else:
        root.handlers[0].setLevel(numeric)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
