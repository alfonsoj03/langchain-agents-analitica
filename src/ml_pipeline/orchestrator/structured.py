from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from ml_pipeline.orchestrator.cursor_llm import ChatCursor
from ml_pipeline.orchestrator.schemas import PipelinePlan

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def parse_json_response(text: str, model: type[T]) -> T:
    stripped = text.strip()
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        stripped = match.group(1).strip()
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    return model.model_validate_json(stripped)


def _augment_messages_for_schema(
    messages: list[BaseMessage], schema: type[BaseModel]
) -> list[BaseMessage]:
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    instruction = (
        "Respond with ONLY valid JSON matching this schema (no markdown, no prose):\n"
        f"{schema_json}"
    )
    augmented = deepcopy(messages)
    for index in range(len(augmented) - 1, -1, -1):
        if isinstance(augmented[index], HumanMessage):
            content = str(augmented[index].content)
            augmented[index] = HumanMessage(content=f"{content}\n\n{instruction}")
            break
    else:
        augmented.append(HumanMessage(content=instruction))
    return augmented


def invoke_structured_plan(llm: ChatCursor, messages: list[BaseMessage]) -> PipelinePlan:
    augmented = _augment_messages_for_schema(messages, PipelinePlan)
    result = llm.invoke(augmented)
    return parse_json_response(str(result.content), PipelinePlan)
