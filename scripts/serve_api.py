#!/usr/bin/env python3
from __future__ import annotations

import socket
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_pipeline.config import get_settings
from ml_pipeline.deployment.api import create_app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> None:
    settings = get_settings()
    model_path = settings.artifacts_dir / "model.pkl"
    port = find_free_port()
    create_app(model_path)
    url = f"http://127.0.0.1:{port}"
    print(f"Serving API at {url}")
    print(f"  Health:  {url}/health")
    print(f"  Predict: {url}/predict")
    uvicorn.run("ml_pipeline.deployment.api:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
