from __future__ import annotations

import logging

from ml_pipeline.logging_config import configure_logging


def test_configure_logging_sets_level():
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO

    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_noisy_loggers_suppressed_at_debug():
    configure_logging("DEBUG")
    assert logging.getLogger("matplotlib").level == logging.WARNING
    assert logging.getLogger("matplotlib.font_manager").level == logging.WARNING
