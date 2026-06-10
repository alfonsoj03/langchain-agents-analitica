from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml_pipeline.config import get_settings
from ml_pipeline.logging_config import configure_logging
from ml_pipeline.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--train", required=True, help="Training CSV path")
    train_parser.add_argument("--future", help="Future/prediction CSV path")
    train_parser.add_argument("--artifacts", help="Artifacts output directory")
    group = train_parser.add_mutually_exclusive_group()
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

    args = parser.parse_args()
    if args.quiet:
        level = "WARNING"
    elif args.verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    configure_logging(level)

    settings = get_settings()

    if args.command == "train":
        pipeline = TrainingPipeline(settings)
        result = pipeline.run(
            train_path=args.train,
            future_path=args.future,
            artifacts_dir=args.artifacts or settings.artifacts_dir,
        )
        logger.info("Prepared dataset: %s", result.prepared_dataset_path)
        logger.info("Model saved: %s", result.model_path)
        logger.info("Raw data report: %s", result.raw_report_path)
        logger.info("Report: %s", result.report_path)
        logger.info("Tree report: %s", result.tree_report_path)
        logger.info("Metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
