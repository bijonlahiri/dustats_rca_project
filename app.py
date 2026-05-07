"""
FastAPI web application for the DU Stats RCA system.

Endpoints:
  GET  /              → Serve the HTML frontend
  POST /api/rca       → Run full agentic RCA workflow from a natural-language query
  GET  /api/health    → Health-check
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

load_dotenv()

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


class RCAResponse(BaseModel):
    site_name: str
    log_date: str
    dominant_rca_label: str
    confidence_score: float
    summary: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: import heavy deps at startup so first request isn't slow
    yield


app = FastAPI(
    title="DU Stats RCA API",
    description="Agentic Root Cause Analysis for telecom network KPIs",
    version="0.1.0",
    lifespan=lifespan,
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


@app.post("/api/rca", response_model=RCAResponse)
async def run_rca(payload: RCARequest):
    """
    Run the full LangGraph agentic RCA workflow from a natural-language query.

    This is a blocking call that may take 30–120 seconds depending on model
    latency and the number of sessions in the data.
    """
    try:
        # Import here to avoid paying startup cost if only health-check is hit
        from rca_agent import run_rca_workflow

        t0 = time.perf_counter()
        # Run in thread pool so FastAPI's event loop stays unblocked
        result: dict[str, Any] = await asyncio.get_event_loop().run_in_executor(
            None, run_rca_workflow, payload.query
        )
        elapsed = round(time.perf_counter() - t0, 2)

        return RCAResponse(
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
