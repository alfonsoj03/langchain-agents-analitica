from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.orchestrator.agent import OrchestratorAgent
from ml_pipeline.orchestrator.schemas import PipelinePlan
from ml_pipeline.preprocessor import Preprocessor
from ml_pipeline.profiling import profile_dataframe
from ml_pipeline.report import generate_html_report
from ml_pipeline.tree_report import generate_tree_html_report
from ml_pipeline.steps.evaluate import evaluate_regression
from ml_pipeline.steps.encode import get_column_kinds
from ml_pipeline.steps.ingest import detect_target, load_csv
from ml_pipeline.steps.persist import ModelBundle, save_bundle
from ml_pipeline.steps.split import split_train_test
from ml_pipeline.steps.train import train_tree_model


@dataclass
class PipelineResult:
    prepared_dataset_path: Path
    model_path: Path
    raw_report_path: Path
    report_path: Path
    tree_report_path: Path
    metrics: dict[str, float]
    best_params: dict[str, Any]
    target: str
    plan: PipelinePlan
    bundle: ModelBundle = field(repr=False)


class TrainingPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        agent: OrchestratorAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.load_yaml()
        self.agent = agent

    def run(
        self,
        train_path: str | Path,
        future_path: str | Path | None = None,
        plan: PipelinePlan | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> PipelineResult:
        pipeline_start = time.perf_counter()
        artifacts = Path(artifacts_dir or self.settings.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)

        split_cfg = self.config.get("split", {})
        tuning_cfg = self.config.get("tuning", {})
        test_size = split_cfg.get("test_size", 0.3)
        random_state = split_cfg.get("random_state", 42)
        n_iter = tuning_cfg.get("n_iter", 20)
        cv = tuning_cfg.get("cv", 3)

        logger.info("=== ML pipeline started ===")
        logger.info("Train data: %s", train_path)
        logger.info("Artifacts: %s", artifacts)
        if future_path:
            logger.info("Future data: %s", future_path)
        logger.info(
            "Config: test_size=%s, random_state=%s, tuning n_iter=%s, cv=%s",
            test_size,
            random_state,
            n_iter,
            cv,
        )

        logger.info("[1/10] Loading data")
        train_df = load_csv(train_path)
        future_df = load_csv(future_path) if future_path else None
        target = detect_target(train_df, future_df)
        logger.info(
            "Loaded training set: %d rows, %d columns; detected target=%s",
            len(train_df),
            len(train_df.columns),
            target,
        )
        if future_df is not None:
            logger.info(
                "Loaded future set: %d rows, %d columns (used for target detection)",
                len(future_df),
                len(future_df.columns),
            )

        logger.info("[2/10] Inferring column types")
        column_kinds = get_column_kinds(train_df, target)
        logger.info("Column types inferred: %s", column_kinds)

        logger.info("[3/10] Generating raw data report")
        raw_report_path = artifacts / "report_raw.html"
        generate_html_report(
            train_df,
            target,
            raw_report_path,
            subtitle="Generated before feature selection and preprocessing",
        )
        logger.info("Raw data report written to %s", raw_report_path)

        logger.info("[4/10] Profiling dataset for orchestrator")
        profile_start = time.perf_counter()
        profile = profile_dataframe(train_df, target=target, column_kinds=column_kinds)
        numeric_count = sum(1 for c in profile["columns"] if c["kind"] == "numeric")
        categorical_count = len(profile["columns"]) - numeric_count
        null_cols = [c["name"] for c in profile["columns"] if c["null_count"] > 0]
        logger.info(
            "Profile complete in %.2fs: %d numeric, %d categorical columns",
            time.perf_counter() - profile_start,
            numeric_count,
            categorical_count,
        )
        if null_cols:
            logger.info("Columns with missing values: %s", ", ".join(null_cols))

        logger.info("[5/10] Planning preprocessing (orchestrator)")
        if plan is None:
            agent = self.agent or OrchestratorAgent(self.settings)
            plan = agent.plan(profile, target)
        else:
            logger.info("Using provided pipeline plan (skipping orchestrator)")

        logger.info("Preprocessing plan:\n%s", plan.model_dump_json(indent=2))

        logger.info("[6/10] Preprocessing: fit + transform")
        preprocess_start = time.perf_counter()
        preprocessor = Preprocessor(target=target, plan=plan, column_kinds=column_kinds)
        prepared = preprocessor.fit_transform(train_df)
        prepared_path = artifacts / "prepared_dataset.csv"
        prepared.to_csv(prepared_path, index=False)
        feature_count = len(prepared.columns) - 1
        logger.info(
            "Preprocessing complete in %.2fs: %d rows x %d features saved to %s",
            time.perf_counter() - preprocess_start,
            len(prepared),
            feature_count,
            prepared_path,
        )

        logger.info("[7/10] Splitting train/test holdout")
        X_train, X_test, y_train, y_test = split_train_test(
            prepared,
            target,
            test_size=test_size,
            random_state=random_state,
        )
        logger.info(
            "Split done: %d train rows, %d test rows (test_size=%s, random_state=%s)",
            len(X_train),
            len(X_test),
            test_size,
            random_state,
        )

        logger.info("[8/10] Training DecisionTree with hyperparameter search")
        tune_start = time.perf_counter()
        model, best_params, tuning_summary = train_tree_model(
            X_train, y_train, self.config
        )
        logger.info(
            "Training complete in %.2fs: best_cv_rmse=%.4f (%d candidates evaluated)",
            time.perf_counter() - tune_start,
            tuning_summary["best_cv_rmse"],
            tuning_summary["n_candidates"],
        )
        logger.info("Best hyperparameters: %s", best_params)

        logger.info("[9/10] Evaluating on holdout set")
        metrics = evaluate_regression(model, X_test, y_test)
        logger.info(
            "Holdout metrics — RMSE: %.4f, MAE: %.4f, MAPE: %.4f, MSE: %.4f",
            metrics["rmse"],
            metrics["mae"],
            metrics["mape"],
            metrics["mse"],
        )

        logger.info("[10/10] Saving artifacts")
        variables = prepared.drop(columns=[target]).columns.values
        bundle = ModelBundle(
            preprocessor=preprocessor,
            model=model,
            variables=variables,
            target=target,
            metrics=metrics,
            best_params=best_params,
            tuning_summary=tuning_summary,
            metadata={"train_rows": len(train_df), "prepared_features": len(variables)},
        )
        model_path = artifacts / "model.pkl"
        save_bundle(bundle, model_path)
        logger.info("Model bundle saved to %s (%d features, %d training rows)", model_path, len(variables), len(train_df))

        report_path = artifacts / "report.html"
        generate_html_report(prepared, target, report_path, metrics=metrics)
        logger.info("Prepared-data report written to %s", report_path)

        tree_report_path = artifacts / "tree.html"
        generate_tree_html_report(
            model,
            feature_names=list(variables),
            target=target,
            output_path=tree_report_path,
            metrics=metrics,
            best_params=best_params,
        )
        logger.info("Tree report written to %s", tree_report_path)

        logger.info("=== Pipeline complete in %.2fs ===", time.perf_counter() - pipeline_start)

        return PipelineResult(
            prepared_dataset_path=prepared_path,
            model_path=model_path,
            raw_report_path=raw_report_path,
            report_path=report_path,
            tree_report_path=tree_report_path,
            metrics=metrics,
            best_params=best_params,
            target=target,
            plan=plan,
            bundle=bundle,
        )
