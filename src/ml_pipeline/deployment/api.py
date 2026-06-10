from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ml_pipeline.config import get_settings
from ml_pipeline.deployment.prepare_future import prepare_and_predict
from ml_pipeline.steps.persist import ModelBundle, load_bundle

app = FastAPI(title="ML Pipeline Prediction API")
_bundle: ModelBundle | None = None
_model_path: Path | None = None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


def get_bundle(model_path: Path | None = None) -> ModelBundle:
    global _bundle, _model_path
    import os

    if model_path is None and _bundle is not None:
        return _bundle

    env_path = os.getenv("MODEL_PATH")
    path = model_path or (Path(env_path) if env_path else get_settings().artifacts_dir / "model.pkl")
    if _bundle is None or _model_path != path:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        _bundle = load_bundle(path)
        _model_path = path
    return _bundle


def create_app(model_path: str | Path | None = None) -> FastAPI:
    if model_path:
        get_bundle(Path(model_path))
    return app


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    future_df = pd.DataFrame(request.records)
    result = prepare_and_predict(bundle, future_df)
    return {"records": result.to_dict(orient="records"), "count": len(result)}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    content = await file.read()
    future_df = pd.read_csv(io.BytesIO(content))
    result = prepare_and_predict(bundle, future_df)
    return {"records": result.to_dict(orient="records"), "count": len(result)}
