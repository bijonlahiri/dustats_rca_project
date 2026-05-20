"""
FastAPI web application for the DU Stats RCA system.

Endpoints:
  GET  /              → Serve the HTML frontend
  POST /api/session   → Create a new conversation thread, returns thread_id
  POST /api/rca       → Run full agentic RCA workflow from a natural-language query
  GET  /api/health    → Health-check
"""

from __future__ import annotations

import asyncio
import time
import uuid
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from logger.logger import logging

load_dotenv()

# ---------------------------------------------------------------------------
# S3 Usage for memory layer
# ---------------------------------------------------------------------------

import boto3
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "./memory")

# Initialize the s3 client

if USE_S3:
    s3_client = boto3.client("s3")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RCARequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        description="Natural-language question about a telecom site.",
        examples=["What is the situation of Nashik site on 1 Jan 2026?"],
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread ID for multi-turn continuity. "
            "Omit (or pass null) to start a new conversation. "
            "Pass the thread_id returned by a prior /api/rca call to ask "
            "follow-up questions that inherit site/date/cell/UE context."
        ),
    )


class SiteResult(BaseModel):
    site_name: str
    dominant_rca_label: str
    confidence_score: float
    summary: str


class RCAResponse(BaseModel):
    thread_id: str
    is_comparison: bool = False
    # Single-site fields (populated when is_comparison=False)
    site_name: str = ""
    dominant_rca_label: str = ""
    confidence_score: float = 0.0
    summary: str = ""
    # Multi-site fields (populated when is_comparison=True)
    sites: list[SiteResult] = []
    comparison_summary: str = ""
    log_date: str
    elapsed_seconds: float


class SessionResponse(BaseModel):
    thread_id: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the inference pipeline and compiled graph at startup.
    # On Lambda, this runs once per container and amortises the cold-start
    # cost of downloading MLflow models and compiling the LangGraph.
    import asyncio as _asyncio
    from rca_agent import _get_pipeline
    await _asyncio.get_event_loop().run_in_executor(None, _get_pipeline)
    yield


app = FastAPI(
    title="DU Stats RCA API",
    description="Agentic Root Cause Analysis for telecom network KPIs",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


# Mount static files only if the directory exists
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse, summary="Create a new conversation thread")
async def create_session():
    """
    Create a new conversation thread and return its ID.

    Use the returned thread_id in subsequent /api/rca calls to maintain
    multi-turn context across follow-up questions.
    """
    return SessionResponse(thread_id=str(uuid.uuid4()))


@app.post("/api/rca", response_model=RCAResponse)
async def run_rca(payload: RCARequest):
    """
    Run the full LangGraph agentic RCA workflow from a natural-language query.

    Pass thread_id from a prior response to ask follow-up questions in the
    same conversation. The agent will inherit site, date, cell, and UE context
    from prior turns and resolve relative references like "the next day" or
    "cell 0 instead".

    This is a blocking call that may take 30–120 seconds depending on model
    latency and the number of sessions in the data.
    """
    try:
        from rca_agent import parse_query, run_rca_ue_comparison, run_rca_workflow

        t0 = time.perf_counter()

        from utils.utils import load_conversations
        thread_id = payload.thread_id
        message_history = load_conversations(thread_id) if thread_id else None

        # Detect comparison queries before running the full workflow.
        # A query triggers UE comparison when either multiple UE IDs are listed
        # explicitly, or no UE ID is given (triggering auto-discovery of all UEs).
        params = await asyncio.get_event_loop().run_in_executor(
            None, parse_query, payload.query, message_history
        )
        logging.info(f"[parese_query] Params:\n{params}")
        ueids = params.get("ueids", [])
        is_comparison = len(ueids) != 1

        if is_comparison:
            logging.info(f"Running RCA UE comparison...")
            result: dict[str, Any] = await asyncio.get_event_loop().run_in_executor(
                None, run_rca_ue_comparison, params, payload.query, payload.thread_id
            )
        else:
            logging.info(f"Running RCA workflow...")
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_rca_workflow, payload.query, payload.thread_id
            )

        elapsed = round(time.perf_counter() - t0, 2)

        if is_comparison:
            return RCAResponse(
                thread_id=result.get("thread_id", ""),
                is_comparison=True,
                log_date=result.get("log_date", ""),
                sites=[
                    SiteResult(
                        site_name=s.get("site_name", ""),
                        dominant_rca_label=s.get("dominant_rca_label", "Unknown"),
                        confidence_score=float(s.get("confidence_score", 0.0)),
                        summary=s.get("summary", ""),
                    )
                    for s in result.get("sites", [])
                ],
                comparison_summary=result.get("comparison_summary", ""),
                elapsed_seconds=elapsed,
            )

        return RCAResponse(
            thread_id=result.get("thread_id", ""),
            is_comparison=False,
            site_name=result.get("site_name", ""),
            log_date=result.get("log_date", ""),
            dominant_rca_label=result.get("dominant_rca_label", "Unknown"),
            confidence_score=float(result.get("confidence_score", 0.0)),
            summary=result.get("summary", ""),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
