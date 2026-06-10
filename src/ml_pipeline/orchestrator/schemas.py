from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ColumnDrop(BaseModel):
    name: str
    reason: str


class OutlierConfig(BaseModel):
    apply: bool = False
    method: Literal["iqr_clip", "none"] = "none"


class PipelinePlan(BaseModel):
    columns_to_drop: list[ColumnDrop] = Field(default_factory=list)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)
    model_family: Literal["tree"] = "tree"
