from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(default).lower()).strip().lower()
    return raw in ("1", "true", "yes")


def _env_reasoning(name: str, default: str = "low") -> str | None:
    raw = os.getenv(name, default).strip().lower()
    if raw in ("", "default", "none", "off"):
        return None
    return raw


@dataclass
class Settings:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )
    cursor_api_key: str = field(default_factory=lambda: os.getenv("CURSOR_API_KEY", ""))
    cursor_model: str = field(
        default_factory=lambda: os.getenv("CURSOR_MODEL", "claude-sonnet-4-6")
    )
    cursor_model_fast: bool = field(
        default_factory=lambda: _env_bool("CURSOR_MODEL_FAST", False)
    )
    cursor_model_reasoning: str | None = field(
        default_factory=lambda: _env_reasoning("CURSOR_MODEL_REASONING", "low")
    )
    project_root: Path = PROJECT_ROOT
    config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "pipeline_config.yaml")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")

    @classmethod
    def load(cls) -> Settings:
        settings = cls()
        if settings.config_path.exists():
            with open(settings.config_path) as f:
                cfg = yaml.safe_load(f) or {}
            artifacts = cfg.get("artifacts_dir", "artifacts")
            settings.artifacts_dir = settings.project_root / artifacts
        return settings

    def load_yaml(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    return Settings.load()
