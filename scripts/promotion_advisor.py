#!/usr/bin/env python3
"""Promotion advisor — fetch ML predictions, then ask Cursor LLM for recommendations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_pipeline.config import get_settings
from ml_pipeline.orchestrator.cursor_llm import get_cursor_llm

DEFAULT_CUSTOMERS_CSV = ROOT / "data" / "videojuegos-datosFuturos.csv"

SYSTEM_PROMPT = """You are a Smart Promotion Advisor for a videogame retail chain.

You receive customer profiles with ML-predicted spending budgets (the "prediccion" field).

Classify each customer:
- Low tier: predicted budget < 150
- Mid tier: predicted budget 150–300
- Premium tier: predicted budget > 300

Recommend promotions by tier:
- Low: send a 20% discount coupon for their platform
- Mid: offer a loyalty points bundle
- Premium: invite to exclusive early-access event

Personalize each recommendation using game, age, sex, platform, and habitual status.
Write a plain-text report with one section per customer: game, predicted budget,
tier, recommended action, and a brief justification.
"""


def get_api_url() -> str:
    url = os.getenv("PREDICTION_API_URL", "").strip().rstrip("/")
    if not url:
        print(
            "ERROR: PREDICTION_API_URL is not set.\n"
            "Start the API with: python scripts/serve_api.py\n"
            "Then set: export PREDICTION_API_URL=http://127.0.0.1:<port>"
        )
        sys.exit(1)
    return url


def verify_api(url: str) -> None:
    try:
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"ERROR: Cannot reach prediction API at {url}\n"
            f"Details: {exc}\n"
            "Make sure `python scripts/serve_api.py` is running and the URL is correct."
        )
        sys.exit(1)


def load_customers(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: Customer file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def fetch_predictions(api_url: str, records: list[dict]) -> list[dict]:
    response = requests.post(
        f"{api_url}/predict",
        json={"records": records},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["records"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ML predictions and ask Cursor LLM for promotion recommendations"
    )
    parser.add_argument(
        "--customers",
        type=Path,
        default=DEFAULT_CUSTOMERS_CSV,
        help=f"CSV with customer profiles (default: {DEFAULT_CUSTOMERS_CSV.name})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.cursor_api_key:
        print("ERROR: CURSOR_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    api_url = get_api_url()
    verify_api(api_url)

    print("Starting promotion advisor", flush=True)

    print(f"1. Loading customers from {args.customers.name}", flush=True)
    customers = load_customers(args.customers)

    print(f"2. Fetching predictions ({len(customers)} records)", flush=True)
    predictions = fetch_predictions(api_url, customers)
    predictions_json = json.dumps(predictions, ensure_ascii=False, indent=2)

    print("3. Generating promotion report with Cursor LLM", flush=True)
    llm = get_cursor_llm(settings)
    llm.agent_name = "Smart Promotion Advisor"
    llm.verbose = False

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Here are today's customers with their ML-predicted budgets:\n\n"
                f"{predictions_json}\n\n"
                "Write the promotion report."
            )
        ),
    ]

    try:
        result = llm.invoke(messages)
        report = str(result.content).strip()
        print("\nOutput:\n", flush=True)
        print(report)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
