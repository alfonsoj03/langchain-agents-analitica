#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_pipeline.config import get_settings
from ml_pipeline.logging_config import configure_logging
from ml_pipeline.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ML training pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and the final summary",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show extra debug output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.quiet:
        level = "WARNING"
    elif args.verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    configure_logging(level)

    settings = get_settings()
    data_dir = settings.project_root / "data"
    pipeline = TrainingPipeline(settings)
    result = pipeline.run(
        train_path=data_dir / "videojuegos.csv",
        future_path=data_dir / "videojuegos-datosFuturos.csv",
        artifacts_dir=settings.artifacts_dir,
    )
    logger.info("Prepared dataset: %s", result.prepared_dataset_path)
    logger.info("Model saved: %s", result.model_path)
    logger.info("Raw data report: %s", result.raw_report_path)
    logger.info("Report: %s", result.report_path)
    logger.info("Tree report: %s", result.tree_report_path)
    logger.info("Metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
