# LangChain Autonomous ML Pipeline

Generalizable regression ML pipeline with a Gemini orchestrator agent (LangChain, not LangGraph) for data-prep decisions and deterministic execution for all mechanical steps.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env locally — use exactly this format (never commit .env):
#   GOOGLE_API_KEY=your_key_here
#   GEMINI_MODEL=gemini-3.5-flash
```

## Run pipeline

```bash
python scripts/run_pipeline.py
```

Or via CLI:

```bash
python -m ml_pipeline.cli train \
  --train data/videojuegos.csv \
  --future data/videojuegos-datosFuturos.csv
```

## Outputs (`artifacts/`)

| File | Description |
|------|-------------|
| `prepared_dataset.csv` | Final training data after prep (traceability) |
| `model.pkl` | Preprocessor + tree model + variables + metrics |
| `report.html` | Statistical description and visualizations |
| `tree.html` | Trained decision tree visualization and text rules |

## Serve API

```bash
python scripts/serve_api.py
```

Endpoints:
- `GET /health`
- `POST /predict` — JSON body `{"records": [{...}, ...]}`
- `POST /predict/csv` — CSV file upload

## Promotion advisor

Fetches predictions from the running API, then uses the Cursor LLM (single prompt, no ReAct) to write personalized promotion recommendations:

```bash
# Terminal 1
python scripts/serve_api.py
# Copy the printed URL into .env as PREDICTION_API_URL

# Terminal 2
python scripts/promotion_advisor.py
```

## Tests

Requires `GOOGLE_API_KEY` in `.env` for live Gemini orchestrator tests.

```bash
pytest tests/unit -v
pytest tests/integration -v
pytest tests/e2e -v -m e2e
pytest -v
```

## Acceptance criteria

- [x] Generalizable pipeline (numeric-only, categorical-only, mixed datasets)
- [x] Orchestrator decides prep steps only; training/tuning is deterministic
- [x] Tree model with hyperparameter search on 70% train, evaluated on 30% holdout
- [x] `prepared_dataset.csv`, `model.pkl`, `report.html` artifacts
- [x] FastAPI prediction service mirroring training prep
- [x] TDD unit tests + integration tests + end-to-end test
