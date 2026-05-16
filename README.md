# DU Stats RCA Project

An agentic machine learning system for automated **Root Cause Analysis (RCA)** of telecom network performance degradation. The system combines a multi-task LSTM neural network with a LangGraph-based agentic AI workflow to diagnose network issues at the cell and UE (User Equipment) level from KPI time-series data.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [RCA Labels](#rca-labels)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training Pipeline](#training-pipeline)
  - [Inference](#inference)
  - [Agentic RCA Workflow](#agentic-rca-workflow)
  - [Web API](#web-api)
- [ML Pipeline Details](#ml-pipeline-details)
  - [Data Ingestion](#data-ingestion)
  - [Data Validation](#data-validation)
  - [Data Transformation](#data-transformation)
  - [Model Architecture](#model-architecture)
  - [Model Training](#model-training)
- [Agentic AI Workflow](#agentic-ai-workflow)
- [API Reference](#api-reference)
- [CI/CD](#cicd)
- [Lambda Performance Optimizations](#lambda-performance-optimizations)
- [Dependencies](#dependencies)

---

## Overview

Telecom networks generate high-frequency KPI streams — metrics like BLER (Block Error Rate), CQI (Channel Quality Indicator), and MCS (Modulation and Coding Scheme) — across thousands of cells and user sessions. Manual RCA of performance degradations is time-consuming and requires deep domain expertise.

This project automates that process with a two-stage approach:

1. **LSTM model** trained on historical labeled sessions to predict the RCA category and the approximate time at which an issue started, given a sequence of 6 KPI features.
2. **Agentic AI layer** (LangGraph + GPT) that retrieves both model predictions and raw KPI signals for a given query, cross-validates them, and generates a confidence-scored, plain-English RCA report.

---

## Architecture

```
Natural Language Query (e.g., "What happened at Nashik on 2026-01-01?")
        │
        ▼
┌───────────────────┐
│  Parse Query Node │  Extracts: site_name, log_date, cellid, ueid
│  (GPT-4o-mini)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│                    Judge Agent (ReAct Loop)            │
│                    (gpt-5-nano)                        │
│                                                        │
│  ┌─────────────────┐      ┌───────────────────────┐   │
│  │ run_inference() │      │     fetch_data()       │   │
│  │                 │      │                        │   │
│  │  InferencePipe  │      │  Databricks SQL →      │   │
│  │  → MLflow load  │      │  30-min KPI windows    │   │
│  │  → LSTM predict │      └───────────────────────┘   │
│  │  → RCA label +  │                                   │
│  │    issue start  │                                   │
│  └─────────────────┘                                   │
│                                                        │
│  Compares predictions vs. raw signal anomalies         │
│  Outputs: { rca_label, confidence: 0.0–1.0, reasoning }│
└────────────────────────────┬──────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Summarizer Agent    │  Plain-English RCA report
                  │  (GPT-4o)            │  ≤200 words
                  └──────────┬───────────┘
                             │
                             ▼
          { site_name, log_date, dominant_rca_label,
            confidence_score, summary }
```

---

## Project Structure

```
dustats_rca_project/
├── .github/
│   └── workflows/
│       └── register_model.yml      # CI/CD: trains & registers model on push to main
├── logger/
│   ├── __init__.py
│   └── logger.py                   # Centralized logging configuration
├── src/
│   ├── components/
│   │   ├── ingestion.py            # Parallel data fetch from Databricks
│   │   ├── validation.py           # Structural, physics, and statistical checks
│   │   ├── transformation.py       # Feature scaling, session windowing, tensor creation
│   │   ├── model.py                # MultiHeadLSTM PyTorch architecture
│   │   ├── model_trainer.py        # Training loop with MLflow logging
│   │   └── model_wrapper.py        # MLflow PyFunc wrapper
│   └── pipelines/
│       ├── training_pipeline.py    # Orchestrates ingestion → validation → transform → train
│       └── inference_pipeline.py   # Lazy-loads models; runs predictions from MLflow
├── utils/
│   └── utils.py                    # DB queries, train/val steps, session processing, metrics
├── app.py                          # FastAPI web server
├── rca_agent.py                    # LangGraph agentic RCA workflow
├── main.py                         # CLI entry point for training
├── inference.py                    # Standalone inference script
├── pyproject.toml                  # Project metadata and dependencies
├── .python-version                 # Python 3.11
└── .env                            # Environment variables (not committed)
```

---

## RCA Labels

The model classifies each session into one of four categories:

| Label | Name | Description |
|-------|------|-------------|
| `0` | No Issue | Normal operation, no degradation detected |
| `1` | High DL BLER — Bad DL Channel Quality | Elevated block error rate caused by poor downlink channel conditions |
| `2` | Static DL BLER — Good DL Channel | Persistent BLER despite healthy channel; likely a configuration or scheduler issue |
| `3` | Scheduler Limited MCS — Good DL Channel | MCS capped by scheduler despite good channel; throughput is artificially constrained |

---

## Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- Access to a Databricks workspace with:
  - SQL warehouse endpoint
  - Unity Catalog enabled
  - Table: `du_stats.training_data.synth_time_series_rca_table`
- OpenAI API key (for the agentic workflow)
- MLflow tracking configured to Databricks

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd dustats_rca_project

# Install dependencies with UV
uv sync

# Or with pip
pip install -e .
```

---

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Databricks SQL Warehouse
DATABRICKS_SERVER_HOSTNAME=<your-workspace>.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
DATABRICKS_ACCESS_TOKEN=<personal-access-token>

# Databricks MLflow Registry
DATABRICKS_HOST=https://<your-workspace>.azuredatabricks.net
DATABRICKS_TOKEN=<personal-access-token>

# OpenAI (required for agentic RCA workflow)
OPENAI_API_KEY=<your-openai-api-key>
```

---

## Usage

### Training Pipeline

Run the full training pipeline (ingestion → validation → transformation → training):

```bash
python main.py \
  --date 2026-01-01 \       # Fetch data logged after this date
  --path artifacts \        # Output directory for intermediate artifacts
  --workers 4 \             # Parallel workers for data ingestion
  --epochs 100              # Training epochs
```

This will:
1. Fetch labeled KPI sessions from Databricks in parallel
2. Validate data quality (structural, RF physics, statistical)
3. Scale features, create session tensors, split 80/20 train/test
4. Train the MultiHeadLSTM and log metrics to MLflow
5. Register the trained model and preprocessor to Databricks Unity Catalog

### Inference

Run predictions on a specific site and date:

```bash
python inference.py \
  --log_date 2026-01-01 \
  --site_name nashik \
  --path output
```

Returns the predicted RCA label and estimated issue start time (seconds into the session).

### Agentic RCA Workflow

Query the system using natural language:

```bash
python rca_agent.py "What is the network situation at the Nashik site on January 1st, 2026?"
```

Or import and call programmatically:

```python
from rca_agent import run_rca_workflow

result = await run_rca_workflow(
    "What happened at Nashik on 2026-01-01?"
)
print(result)
# {
#   "site_name": "nashik",
#   "log_date": "2026-01-01",
#   "dominant_rca_label": 1,
#   "confidence_score": 0.87,
#   "summary": "The Nashik site experienced elevated downlink BLER..."
# }
```

### Web API

Start the FastAPI server:

```bash
python app.py
# Server runs at http://0.0.0.0:8000
```

Open `http://localhost:8000` in a browser for the HTML frontend, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/rca \
  -H "Content-Type: application/json" \
  -d '{"query": "What happened at Nashik on 2026-01-01?"}'
```

---

## ML Pipeline Details

### Data Ingestion

- **Source:** Databricks SQL table `du_stats.training_data.synth_time_series_rca_table`
- Fetches data after a configurable `log_date` cutoff
- Parallelized with `ThreadPoolExecutor` for fast multi-site extraction
- Outputs one Parquet file per `(site_name, log_date)` pair

### Data Validation

The `Validation` component performs three layers of checks before any data reaches the model:

**Structural integrity:**
- Time window format and completeness
- BLER / IBLER / TBLER bounds: must be in `[0, 100]`
- Tick count consistency for LSTM sequence inputs

**RF physics validation:**
- Flags physically impossible combinations (e.g., high CQI with consistently low MCS)
- Detects anomalous correlations between channel quality indicators

**Statistical validation:**
- Kruskal-Wallis H-test: confirms each KPI feature differs significantly across RCA label groups
- LSTM readiness: checks sequence continuity at 30-second resolution

Outputs a JSON validation report and a validated Parquet dataset partitioned by `log_date` and `site_name`.

### Data Transformation

The `TelecomGridTransformer` prepares data for LSTM input:

| Step | Detail |
|------|--------|
| Feature selection | `cqi`, `mcs`, `ibler`, `rbler`, `resbler`, `tbler` |
| Scaling | `StandardScaler` per feature (sklearn `ColumnTransformer`) |
| Session reindexing | Uniform 30-second grid, max uptime 28,770 s (~480 min) |
| Windowing | Fixed-length sequences: `seq_len = 960` samples = 480 minutes |
| Train/test split | 80/20 stratified by RCA label |
| Output format | PyTorch `.pth` tensors + `preprocessor.pkl` (joblib) |

### Model Architecture

**MultiHeadLSTM** (`src/components/model.py`):

```
Input:  (batch, seq_len=960, d_in=6)
  │
  ├─ Linear projection:  d_in → d_out
  │
  ├─ LSTM:  d_out hidden, 3 layers  [optionally bidirectional]
  │
  ├─ Mean pooling over sequence dimension
  │
  ├─ Dense (d_out → d_out) + ReLU
  │
  ├─ issue_start_head   →  scalar ∈ [0, 1]  (normalized start time)
  └─ rca_label_head     →  4 logits          (RCA category)
```

**Multi-task loss:**

```
Loss = w₁ · MSE(predicted_start, true_start) + w₂ · CrossEntropy(predicted_label, true_label)
```

Class imbalance is addressed with inverse class-frequency weights applied to the CrossEntropy term.

### Model Training

- **Optimizer:** Adam (`lr = 1e-4`)
- **MLflow logging:** per-class F1 score, precision, and confusion matrix — logged every 10 epochs
- **Registered artifacts** in Databricks Unity Catalog:
  - `multi_head_lstm_telecom_model` — PyTorch model checkpoint
  - `preprocessor_model` — sklearn ColumnTransformer (joblib)

---

## Agentic AI Workflow

The `rca_agent.py` module implements a LangGraph state graph with four nodes:

| Node | Model | Role |
|------|-------|------|
| `parse_query` | GPT-4o-mini | Extracts structured parameters from natural language |
| `judge_agent` | gpt-5-nano | ReAct loop — calls tools, cross-validates signals, scores confidence |
| `tool_node` | — | Executes `run_inference()` and `fetch_data()` tools |
| `summarizer_agent` | GPT-4o | Writes the final plain-English RCA report (≤200 words) |

**Graph flow:**

```
parse_query → judge_agent ⟷ tool_node → summarizer_agent → END
```

**Judge Agent tools:**

- `run_inference(site_name, log_date, cellid?, ueid?)` — Loads LSTM from the MLflow registry, returns the predicted RCA label and issue start time
- `fetch_data(site_name, log_date)` — Queries Databricks for 30-minute windowed KPI averages for the same period

The judge compares LSTM predictions against raw signal anomalies and outputs a `confidence_score` between 0.0 and 1.0 before handing off to the summarizer.

---

## API Reference

### `POST /api/rca`

Run agentic RCA for a natural language query.

**Request body:**
```json
{
  "query": "string"
}
```

**Response:**
```json
{
  "site_name": "string",
  "log_date": "YYYY-MM-DD",
  "dominant_rca_label": 1,
  "confidence_score": 0.87,
  "summary": "string"
}
```

### `GET /api/health`

Returns `{"status": "ok"}` when the server is running.

### `GET /`

Serves the HTML frontend for interactive RCA queries.

---

## CI/CD

The GitHub Actions workflow at [.github/workflows/register_model.yml](.github/workflows/register_model.yml) triggers on pushes to `main` that modify `src/`, `utils/`, `main.py`, or `pyproject.toml`.

Pipeline steps:
1. Set up Python 3.11 and install UV
2. Sync all project dependencies
3. Run the full training pipeline (`--date 1970-01-01`, 100 epochs, 32 workers)
4. Register the new model version to Databricks Unity Catalog

Databricks credentials are supplied as GitHub Actions secrets.

---

## Lambda Performance Optimizations

The following optimizations were applied to reduce latency on AWS Lambda relative to local execution.

### 1. Compiled LangGraph reused across requests

**Problem:** `build_graph()` was called inside `run_rca_workflow()`, recompiling the entire LangGraph state machine on every `/api/rca` request.

**Fix:** The graph is now compiled once at module load time into the module-level `_COMPILED_GRAPH` constant and reused for every subsequent call.

```python
# rca_agent.py
_COMPILED_GRAPH = _build_graph()   # compiled once when the module is imported

def run_rca_workflow(natural_query: str) -> dict:
    app = _COMPILED_GRAPH           # no recompilation per request
    ...
```

### 2. Inference pipeline pre-warmed at container startup

**Problem:** On the first `/api/rca` request, the Lambda container had to download both MLflow models from Databricks (preprocessor + LSTM), adding several seconds of latency to that request.

**Fix:** The FastAPI `lifespan` handler now eagerly calls `_get_pipeline()` at startup, so model downloads happen once when the container initialises, not on the first user request.

```python
# app.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    from rca_agent import _get_pipeline
    await asyncio.get_event_loop().run_in_executor(None, _get_pipeline)
    yield
```

### 3. Databricks SQL query deduplicated with a fetch cache

**Problem:** Within a single RCA workflow the `run_inference` tool and the `fetch_data` tool both called `pipeline.fetch_data()` with identical arguments, resulting in two round-trip SQL queries to Databricks for the same data.

**Fix:** A lightweight single-entry cache (`_fetch_cache`) intercepts repeated calls with the same `(site_name, log_date, cellid, ueid)` key and returns the already-fetched DataFrame. The cache is cleared before storing a new key so memory stays bounded.

```python
# rca_agent.py
_fetch_cache: dict = {}

def _cached_fetch(site_name, log_date, cellid, ueid):
    key = (site_name.lower(), log_date, cellid, ueid)
    if key not in _fetch_cache:
        _fetch_cache.clear()          # bound to one entry
        _fetch_cache[key] = _get_pipeline().fetch_data(...)
    return _fetch_cache[key]
```

Both `run_inference` and `fetch_data` tools now call `_cached_fetch` instead of `pipeline.fetch_data` directly, cutting the number of Databricks SQL queries per request from 2 to 1.

---

## Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| Deep Learning | `torch<2.3` | LSTM model, tensors, training loop |
| ML | `scikit-learn>=1.8.0` | Feature scaling, metrics |
| Data | `pandas<2.3`, `numpy<2.0`, `pyarrow` | Data manipulation and Parquet I/O |
| Statistics | `scipy>=1.17.1` | Kruskal-Wallis statistical test |
| Data Warehouse | `databricks-sql-connector[pyarrow]>=4.2.6` | Databricks SQL access |
| LLM | `langchain-openai>=0.1.0` | OpenAI API integration for agents |
| Agentic | `langgraph>=0.2.0` | Multi-agent state graph |
| MLOps | `mlflow[databricks]>=3.11.1` | Experiment tracking and model registry |
| Web | `fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0` | REST API server |
| Visualization | `matplotlib>=3.9.0`, `seaborn>=0.13.0` | Plots and confusion matrix |
| Config | `python-dotenv>=1.2.2`, `pyaml>=26.2.1` | Environment and config loading |
