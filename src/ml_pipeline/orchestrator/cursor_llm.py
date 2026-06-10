"""LangChain chat model backed by the Cursor Cloud Agents API."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr, SecretStr

if TYPE_CHECKING:
    from ml_pipeline.config import Settings

CURSOR_API_BASE = "https://api.cursor.com"
TERMINAL_RUN_STATUSES = frozenset({"FINISHED", "ERROR", "CANCELLED", "EXPIRED"})
ACTIVE_RUN_STATUSES = frozenset({"CREATING", "RUNNING"})
AGENT_NAME = "ML Pipeline Orchestrator"

_STORAGE_MODE_HINT = """
Cursor Cloud Agents API requires "storage mode" (Privacy Mode with storage).

Why: Each API agent is a durable cloud session. Cursor must store conversation
state, run history, and workspace metadata on their servers. That is blocked
when you use Privacy Mode (Legacy) or Ghost / no-storage mode.

How to fix:
  1. Open https://cursor.com/settings (or Cursor app: Cmd+Shift+J → General).
  2. Switch privacy from "Privacy Mode (Legacy)" to "Privacy Mode".
     (Also called "Privacy Mode with storage" in some builds.)
  3. If you use Ghost / Local mode: Settings → Advanced → turn Ghost mode OFF.
  4. On a team: an admin may need to change Dashboard → Settings → Privacy.
  5. Restart the Cursor app after changing the setting.

Your code is still not used for model training; see https://cursor.com/data-use
"""


def _format_api_error(status_code: int, detail: object) -> str:
    if isinstance(detail, dict):
        err = detail.get("error", detail)
        if isinstance(err, dict):
            code = err.get("code", "")
            message = err.get("message", str(detail))
            if code == "feature_unavailable" and "storage" in message.lower():
                return f"Cursor API failed ({status_code}): {message}{_STORAGE_MODE_HINT}"
            return f"Cursor API failed ({status_code}): {message}"
    return f"Cursor API failed ({status_code}): {detail}"


_MODELS_CATALOG: list[dict[str, Any]] | None = None


def _load_models_catalog(session: requests.Session, api_base: str) -> list[dict[str, Any]]:
    global _MODELS_CATALOG
    if _MODELS_CATALOG is not None:
        return _MODELS_CATALOG
    url = f"{api_base.rstrip('/')}/v1/models"
    response = session.get(url, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to load Cursor models catalog ({response.status_code})")
    _MODELS_CATALOG = response.json().get("items", [])
    return _MODELS_CATALOG


def _resolve_model_params(
    session: requests.Session,
    api_base: str,
    model_id: str,
    *,
    model_fast: bool,
    model_reasoning: str | None,
) -> list[dict[str, str]]:
    """Pick a valid variant from GET /v1/models (required by the agents API)."""
    catalog = _load_models_catalog(session, api_base)
    model = next((item for item in catalog if item.get("id") == model_id), None)
    if model is None:
        raise ValueError(
            f"Unknown CURSOR_MODEL={model_id!r}. Call GET /v1/models for valid ids."
        )

    param_defs = {p["id"]: p for p in model.get("parameters", [])}
    desired: dict[str, str] = {}

    if "fast" in param_defs:
        desired["fast"] = "true" if model_fast else "false"
    if model_reasoning:
        if "reasoning" in param_defs:
            desired["reasoning"] = model_reasoning
        elif "effort" in param_defs:
            desired["effort"] = model_reasoning

    variants = model.get("variants", [])
    if desired:
        for variant in variants:
            variant_params = {
                p["id"]: p["value"] for p in variant.get("params", [])
            }
            if all(variant_params.get(k) == v for k, v in desired.items()):
                return [
                    {"id": key, "value": value}
                    for key, value in variant_params.items()
                    if key in desired
                ]

    if model_id.startswith("composer-") and "fast" in param_defs and not model_fast:
        return [{"id": "fast", "value": "false"}]

    if desired:
        return [{"id": key, "value": value} for key, value in desired.items()]

    for variant in variants:
        if variant.get("isDefault"):
            return [
                {"id": p["id"], "value": p["value"]}
                for p in variant.get("params", [])
            ]
    return []


def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            parts.append(f"System:\n{message.content}")
        elif isinstance(message, HumanMessage):
            parts.append(str(message.content))
        elif isinstance(message, AIMessage):
            parts.append(str(message.content))
        else:
            parts.append(str(message.content))
    return "\n\n".join(parts).strip()


class ChatCursor(BaseChatModel):
    """Chat model that runs prompts via Cursor Cloud Agents (no repository)."""

    api_key: SecretStr
    model_id: str = "claude-sonnet-4-6"
    model_fast: bool = False
    model_reasoning: str | None = "low"
    api_base: str = CURSOR_API_BASE
    poll_interval_seconds: float = 2.0
    max_wait_seconds: float = 300.0
    agent_name: str = AGENT_NAME
    verbose: bool = False
    max_llm_calls: int | None = None

    _agent_id: str | None = PrivateAttr(default=None)
    _session: requests.Session | None = PrivateAttr(default=None)
    _llm_call_count: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "cursor-cloud-agent"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_fast": self.model_fast,
            "model_reasoning": self.model_reasoning,
            "api_base": self.api_base,
        }

    def _auth(self) -> requests.auth.AuthBase:
        return requests.auth.HTTPBasicAuth(self.api_key.get_secret_value(), "")

    def _session_client(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.auth = self._auth()
            self._session.headers["Content-Type"] = "application/json"
        return self._session

    def _model_payload(self) -> dict[str, Any]:
        params = _resolve_model_params(
            self._session_client(),
            self.api_base,
            self.model_id,
            model_fast=self.model_fast,
            model_reasoning=self.model_reasoning,
        )
        body: dict[str, Any] = {"id": self.model_id}
        if params:
            body["params"] = params
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        max_retries: int = 5,
    ) -> dict[str, Any]:
        url = f"{self.api_base.rstrip('/')}{path}"
        session = self._session_client()
        for attempt in range(max_retries):
            response = session.request(method, url, json=json, timeout=120)
            if response.status_code == 429:
                retry_after = 30.0
                try:
                    details = response.json()
                    message = details.get("error", {}).get("message", "")
                    if "retry in" in message.lower():
                        match = re.search(r"retry in ([\d.]+)s", message, re.I)
                        if match:
                            retry_after = float(match.group(1)) + 1.0
                except (ValueError, TypeError):
                    pass
                time.sleep(min(retry_after, 60.0))
                continue
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                raise RuntimeError(_format_api_error(response.status_code, detail))
            if response.status_code == 204:
                return {}
            return response.json()

        raise RuntimeError(f"Cursor API {method} {path} failed after retries")

    def _create_run(self, agent_id: str, prompt_text: str) -> str:
        for attempt in range(10):
            try:
                data = self._request(
                    "POST",
                    f"/v1/agents/{agent_id}/runs",
                    json={"prompt": {"text": prompt_text}},
                )
                return data["run"]["id"]
            except RuntimeError as exc:
                if "409" in str(exc) and attempt < 9:
                    time.sleep(self.poll_interval_seconds)
                    continue
                raise
        raise RuntimeError("Unable to create Cursor agent run")

    def _log(self, message: str) -> None:
        if self.verbose:
            preview = message if len(message) <= 500 else message[:500] + "…"
            print(f"[cursor-llm] {preview}", flush=True)

    def _run_prompt(self, prompt_text: str) -> str:
        self._llm_call_count += 1
        call_no = self._llm_call_count
        if self.max_llm_calls is not None and call_no > self.max_llm_calls:
            raise RuntimeError(
                f"Cursor LLM call limit reached ({self.max_llm_calls}). "
                "ReAct agents need one cloud run per Thought/Action step; "
                "raise max_llm_calls or batch work into fewer steps."
            )
        self._log(
            f"LLM call #{call_no}"
            + (f"/{self.max_llm_calls}" if self.max_llm_calls else "")
            + f" — prompt {len(prompt_text)} chars"
        )
        if self._agent_id is None:
            data = self._request(
                "POST",
                "/v1/agents",
                json={
                    "prompt": {"text": prompt_text},
                    "model": self._model_payload(),
                    "name": self.agent_name,
                },
            )
            self._agent_id = data["agent"]["id"]
            run_id = data["run"]["id"]
            self._log(f"Created agent {self._agent_id}, initial run {run_id}")
        else:
            run_id = self._create_run(self._agent_id, prompt_text)
            self._log(f"Reusing agent {self._agent_id}, new run {run_id}")
        result = self._wait_for_run(self._agent_id, run_id)
        self._log(f"Run {run_id} finished — response {len(result)} chars")
        return result

    def _wait_for_run(self, agent_id: str, run_id: str) -> str:
        deadline = time.monotonic() + self.max_wait_seconds
        while time.monotonic() < deadline:
            data = self._request(
                "GET",
                f"/v1/agents/{agent_id}/runs/{run_id}",
            )
            status = data.get("status", "")
            self._log(f"Polling run {run_id}: status={status}")
            if status in TERMINAL_RUN_STATUSES:
                if status == "FINISHED":
                    result = data.get("result") or ""
                    if not result:
                        raise RuntimeError(
                            f"Cursor run {run_id} finished without result text"
                        )
                    return result
                raise RuntimeError(
                    f"Cursor run {run_id} ended with status {status}"
                )
            if status not in ACTIVE_RUN_STATUSES:
                raise RuntimeError(f"Cursor run {run_id} has unknown status {status}")
            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(
            f"Cursor run {run_id} did not finish within {self.max_wait_seconds}s"
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager, kwargs
        prompt_text = _messages_to_prompt(messages)
        if stop:
            prompt_text += (
                "\n\nDo not generate text after any of these stop sequences: "
                + ", ".join(stop)
            )

        text = self._run_prompt(prompt_text)

        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token, 1)[0]

        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def close(self) -> None:
        if self._agent_id:
            try:
                self._request("POST", f"/v1/agents/{self._agent_id}/archive")
            except RuntimeError:
                pass
            self._agent_id = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def get_cursor_llm(settings: Settings | None = None) -> ChatCursor:
    from ml_pipeline.config import get_settings

    cfg = settings or get_settings()
    if not cfg.cursor_api_key:
        raise ValueError(
            "CURSOR_API_KEY is not set. Copy .env.example to .env and add your API key."
        )
    return ChatCursor(
        api_key=SecretStr(cfg.cursor_api_key),
        model_id=cfg.cursor_model,
        model_fast=cfg.cursor_model_fast,
        model_reasoning=cfg.cursor_model_reasoning,
    )
