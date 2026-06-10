from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from ml_pipeline.orchestrator.schemas import PipelinePlan
from ml_pipeline.profiling import profile_to_json

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set. Add it to your local .env file.")
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,
            google_api_key=self.settings.google_api_key,
            temperature=0,
        )
        self.structured_llm = self.llm.with_structured_output(PipelinePlan)

    def plan(self, profile: dict, target: str) -> PipelinePlan:
        profile_json = profile_to_json(profile)
        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Target column: {target}\n\n"
                    f"Dataset profile:\n{profile_json}\n\n"
                    "Return the PipelinePlan."
                )
            ),
        ]
        logger.info(
            "Requesting preprocessing plan from Gemini (%s)…",
            self.settings.gemini_model,
        )
        start = time.perf_counter()
        plan = self.structured_llm.invoke(messages)
        elapsed = time.perf_counter() - start
        plan = self._validate_plan(plan, profile, target)
        outlier_label = plan.outliers.method if plan.outliers.apply else "none"
        logger.info(
            "Orchestrator plan received in %.2fs: drop %d col(s), outliers=%s",
            elapsed,
            len(plan.columns_to_drop),
            outlier_label,
        )
        if plan.columns_to_drop:
            drops = "; ".join(f"{d.name} ({d.reason})" for d in plan.columns_to_drop)
            logger.info("Planned drops: %s", drops)
        return plan

    def _validate_plan(
        self, plan: PipelinePlan, profile: dict, target: str
    ) -> PipelinePlan:
        valid_cols = {c["name"] for c in profile["columns"]}
        plan.columns_to_drop = [
            d for d in plan.columns_to_drop if d.name in valid_cols and d.name != target
        ]
        return plan
