"""
Agentic RCA workflow built with LangGraph.

Graph structure:
  [parse_query] → [judge_agent] ⇄ [tool_node] → [summarizer_agent] → END

  - judge_agent  (Agent 1) : ReAct loop with two tools:
        run_inference  – runs the LSTM RCA model
        fetch_data     – retrieves raw KPI time-series rows
      The LLM compares the two and emits a confidence score + reasoning as JSON.

  - summarizer_agent (Agent 2) : takes the judge JSON and produces a
      plain-English RCA summary.

Usage:
    python rca_agent.py "What is the situation of Nashik site on 1 Jan 2026?"
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Annotated, Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.pipelines.inference_pipeline import InferencePipeline
from logger.logger import logging

load_dotenv()

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

RCA_LABELS: dict[int, str] = {
    0: "No Issue",
    1: "High DL BLER due to bad DL channel quality",
    2: "Static DL BLER, good DL channel",
    3: "Scheduler limited MCS, good DL channel",
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "cqi":    "Channel Quality Indicator — higher is better (0-15)",
    "mcs":    "Modulation and Coding Scheme — higher means faster throughput",
    "ibler":  "Initial Block Error Rate — lower is better",
    "rbler":  "Re-transmission Block Error Rate — lower is better",
    "resbler":"Residual after HARQ Block Error Rate — lower is better",
    "tbler":  "Total Block Error Rate — lower is better",
}

# ---------------------------------------------------------------------------
# Lazy InferencePipeline singleton (avoids re-loading models per tool call)
# ---------------------------------------------------------------------------

_pipeline: InferencePipeline | None = None


def _get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# LangChain tools (decorated with @tool so LangGraph ToolNode can run them)
# ---------------------------------------------------------------------------

@tool
def run_inference(
    site_name: str,
    log_date: str,
    cellid: int | None = None,
    ueid: int | None = None,
) -> str:
    """Run the telecom RCA LSTM inference model for a given site and date.

    Returns predicted RCA category and estimated issue start time per UE session.
    Always call this tool first before fetch_data.

    Args:
        site_name: Telecom site name, e.g. 'nashik'.
        log_date:  Date in YYYY-MM-DD format.
        cellid:    Optional cell ID to filter results.
        ueid:      Optional UE ID to filter results.
    """
    logging.info(f"[Tool] run_inference site={site_name} date={log_date} cell={cellid} ue={ueid}")
    pipeline = _get_pipeline()
    raw_df = pipeline.fetch_data(
        log_date=log_date,
        site_name=site_name.lower(),
        cellid=cellid,
        ueid=ueid,
    )
    df: pd.DataFrame | None = pipeline.predict(raw_df) if raw_df is not None and not raw_df.empty else None
    if df is None or df.empty:
        return json.dumps({"error": "Inference returned no data. Verify site_name and log_date."})

    records = []
    for idx, row in df.iterrows():
        rca_id = int(row["predicted_rca"])
        records.append({
            "site_name":                 idx[0] if isinstance(idx, tuple) else site_name,
            "log_date":                  str(idx[1]) if isinstance(idx, tuple) else log_date,
            "cellid":                    int(idx[2]) if isinstance(idx, tuple) else cellid,
            "ueid":                      int(idx[3]) if isinstance(idx, tuple) else ueid,
            "predicted_issue_start_sec": int(row["predicted_issue_start"]),
            "predicted_rca_id":          rca_id,
            "predicted_rca_label":       RCA_LABELS.get(rca_id, "Unknown"),
            "actual_rca_label":          RCA_LABELS.get(int(row["rca_label"]), "Unknown")
                                         if "rca_label" in row.index else None,
        })

    result = {
        "num_sessions":  len(records),
        "sessions":      records,
        "rca_label_key": RCA_LABELS,
    }
    return json.dumps(result)


@tool
def fetch_data(
    site_name: str,
    log_date: str,
    cellid: int | None = None,
    ueid: int | None = None,
) -> str:
    """Fetch KPI time-series data (CQI, MCS, BLER metrics) aggregated into 30-minute windows.

    Returns per-session windowed KPI means so you can identify when signal quality
    degraded over the course of the session. Use this to verify inference results
    against the observed KPI trends.
    Always call run_inference before calling this tool.

    Args:
        site_name: Telecom site name, e.g. 'nashik'.
        log_date:  Date in YYYY-MM-DD format.
        cellid:    Optional cell ID to filter results.
        ueid:      Optional UE ID to filter results.
    """
    logging.info(f"[Tool] fetch_data site={site_name} date={log_date} cell={cellid} ue={ueid}")
    pipeline = _get_pipeline()
    df: pd.DataFrame | None = pipeline.fetch_data(
        log_date=log_date,
        site_name=site_name.lower(),
        cellid=cellid,
        ueid=ueid,
    )
    if df is None or df.empty:
        return json.dumps({"error": "fetch_data returned no rows. Verify site_name and log_date."})

    kpi_cols = [c for c in ["cqi", "mcs", "ibler", "rbler", "resbler", "tbler"] if c in df.columns]
    window_size = 60  # 60 × 30s = 30 minutes

    session_windows: list[dict] = []
    for (cell, ue), grp in df.groupby(["cellid", "ueid"]):
        grp = grp.sort_values("uptime").reset_index(drop=True)
        grp["window"] = grp["uptime"] // (window_size * 30)  # 30s resolution → seconds
        windowed = (
            grp.groupby("window")[kpi_cols]
            .mean()
            .round(4)
            .reset_index()
        )
        windowed["window_start_sec"] = windowed["window"] * window_size * 30
        windowed = windowed.drop(columns=["window"])
        session_windows.append({
            "cellid":   int(cell),
            "ueid":     int(ue),
            "windows":  windowed.to_dict(orient="records"),
        })

    result = {
        "num_sessions":        len(session_windows),
        "window_duration_sec": window_size * 30,
        "feature_definitions": FEATURE_DESCRIPTIONS,
        "sessions":            session_windows,
    }
    return json.dumps(result)


JUDGE_TOOLS = [run_inference, fetch_data]

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class RCAState(TypedDict):
    # Shared message list (LangGraph reducer keeps full history)
    messages: Annotated[list, add_messages]
    # Structured outputs written by each agent
    site_name:        str
    log_date:         str
    cellid:           int | None
    ueid:             int | None
    judge_json:       dict   # emitted by judge_agent
    rca_summary:      str    # emitted by summarizer_agent
    confidence_score: float


# ---------------------------------------------------------------------------
# Agent 1: LLM-as-a-judge  (ReAct loop via LangGraph)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a telecom network reliability analyst acting as an LLM judge.

Workflow you must follow:
1. Call `run_inference` with the site_name and log_date provided.
2. Call `fetch_data` with the same arguments to obtain the raw KPI time-series rows.
3. Compare the two results:
   - Do the predicted RCA categories match the KPI signals in the raw data?
   - High BLER values (ibler/rbler/resbler/tbler > 0.1) indicate link errors.
   - Low CQI (< 6) suggests interference or poor channel conditions.
   - Low MCS (< 10) indicates degraded modulation — often a hardware or link issue.
4. After calling both tools, output ONLY a JSON object (no markdown, no extra prose):
{
  "inference_summary": {
    "num_sessions": <int>,
    "dominant_rca_label": <str>,
    "sessions_with_issues": <int>,
    "sessions": [<list of session prediction dicts>]
  },
  "raw_kpi_summary": {
    "num_sessions": <int>,
    "kpi_anomalies": [<list of strings describing notable KPI anomalies found in the raw data>]
  },
  "confidence_score": <float 0.0-1.0>,
  "confidence_reasoning": "<2-4 sentences explaining why you assigned this score>",
  "dominant_rca_label": "<most frequent predicted RCA label>",
  "sessions_with_issues": <int, sessions where predicted_rca_id != 0>
}"""


def judge_agent_node(state: RCAState) -> dict:
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0).bind_tools(JUDGE_TOOLS)

    # Prepend system + user prompt on first entry (before any tool calls)
    if not any(isinstance(m, SystemMessage) for m in state["messages"]):
        system_msg = SystemMessage(content=JUDGE_SYSTEM)
        user_msg = HumanMessage(
            content=(
                f"Analyse telecom site '{state['site_name']}' for log_date '{state['log_date']}'"
                + (f", cellid={state['cellid']}" if state.get("cellid") is not None else "")
                + (f", ueid={state['ueid']}" if state.get("ueid") is not None else "")
                + ". Call run_inference first, then fetch_data, then output your JSON."
            )
        )
        messages_in = [system_msg, user_msg] + state["messages"]
    else:
        messages_in = state["messages"]

    response: AIMessage = llm.invoke(messages_in)
    return {"messages": [response]}


def should_continue(state: RCAState) -> str:
    """Route to tools if there are pending tool calls, else move to summarizer."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "summarizer"


def _extract_judge_json(state: RCAState) -> dict:
    """Parse the judge JSON from the last AIMessage."""
    last = state["messages"][-1]
    raw = last.content if isinstance(last, AIMessage) else ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group()) if match else {}


# ---------------------------------------------------------------------------
# Agent 2: RCA Summarizer
# ---------------------------------------------------------------------------

SUMMARIZER_SYSTEM = """You are a senior telecom network operations engineer.
Write a concise Root Cause Analysis (RCA) report for the network operations team
based on the structured analysis you receive.

Required format:
1. One sentence: site name, date, and overall health status.
2. What the ML model detected — RCA category, number of affected sessions, estimated issue start time in seconds into the session.
3. What the raw KPI data shows — cite specific anomalies with numbers from the raw data.
4. Root cause conclusion.
5. Recommended corrective action.
6. Confidence: state the score as a percentage and briefly explain what it means.

Keep the report under 200 words. Plain English only — no JSON, no code blocks."""


def summarizer_agent_node(state: RCAState) -> dict:
    judge_data = _extract_judge_json(state)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    context = (
        f"Site: {state['site_name']}\n"
        f"Log Date: {state['log_date']}\n"
        f"Dominant RCA: {judge_data.get('dominant_rca_label', 'Unknown')}\n"
        f"Sessions with Issues: {judge_data.get('sessions_with_issues', 'N/A')}\n"
        f"Confidence Score: {judge_data.get('confidence_score', 0.0):.2f} / 1.00\n"
        f"Confidence Reasoning: {judge_data.get('confidence_reasoning', '')}\n\n"
        f"Inference Summary:\n{json.dumps(judge_data.get('inference_summary', {}), indent=2)}\n\n"
        f"Raw KPI Anomalies:\n{json.dumps(judge_data.get('raw_kpi_summary', {}), indent=2)}"
    )

    response: AIMessage = llm.invoke([
        SystemMessage(content=SUMMARIZER_SYSTEM),
        HumanMessage(content=context),
    ])

    return {
        "judge_json":       judge_data,
        "confidence_score": float(judge_data.get("confidence_score", 0.0)),
        "rca_summary":      response.content.strip(),
        "messages":         [response],
    }


# ---------------------------------------------------------------------------
# Query parser node
# ---------------------------------------------------------------------------

PARSER_SYSTEM = """Extract telecom query parameters from a natural-language question.
Return ONLY a JSON object with these keys:
  site_name (str, lowercase),
  log_date  (str, YYYY-MM-DD),
  cellid    (int or null),
  ueid      (int or null)

Examples:
  "What is the situation of Nashik site on 1 Jan 2026?"
  -> {"site_name": "nashik", "log_date": "2026-01-01", "cellid": null, "ueid": null}

  "Mumbai cell 3 RCA on 15 March 2025"
  -> {"site_name": "mumbai", "log_date": "2025-03-15", "cellid": 3, "ueid": null}

  "What is the situation of Nashik site on 1 Jan 2026 for cell 0 ueid 17023"
  -> {"site_name": "nashik", "log_date": "2025-01-01", "cellid": 0, "ueid": 17023}
  """



def parse_query_node(state: RCAState) -> dict:
    """Extract structured query params from the initial HumanMessage."""
    query = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            query = m.content
            break

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        SystemMessage(content=PARSER_SYSTEM),
        HumanMessage(content=query),
    ])
    try:
        params = json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        params = json.loads(match.group()) if match else {}

    return {
        "site_name": params.get("site_name", ""),
        "log_date":  params.get("log_date", ""),
        "cellid":    params.get("cellid"),
        "ueid":      params.get("ueid"),
    }


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_graph():
    tool_node = ToolNode(JUDGE_TOOLS)

    graph = StateGraph(RCAState)

    graph.add_node("parse_query", parse_query_node)
    graph.add_node("judge_agent", judge_agent_node)
    graph.add_node("tools",       tool_node)
    graph.add_node("summarizer",  summarizer_agent_node)

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "judge_agent")
    graph.add_conditional_edges(
        "judge_agent",
        should_continue,
        {"tools": "tools", "summarizer": "summarizer"},
    )
    graph.add_edge("tools",      "judge_agent")  # loop back after each tool call
    graph.add_edge("summarizer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rca_workflow(natural_query: str) -> dict[str, Any]:
    """
    Run the full agentic RCA workflow from a natural-language query.

    Args:
        natural_query: e.g. "What is the situation of Nashik site on 1 Jan 2026?"

    Returns:
        dict with keys: site_name, log_date, dominant_rca_label,
                        confidence_score, summary
    """
    app = build_graph()

    initial_state: RCAState = {
        "messages":         [HumanMessage(content=natural_query)],
        "site_name":        "",
        "log_date":         "",
        "cellid":           None,
        "ueid":             None,
        "judge_json":       {},
        "rca_summary":      "",
        "confidence_score": 0.0,
    }

    final_state = app.invoke(initial_state)

    return {
        "site_name":          final_state["site_name"],
        "log_date":           final_state["log_date"],
        "dominant_rca_label": final_state["judge_json"].get("dominant_rca_label", "Unknown"),
        "confidence_score":   final_state["confidence_score"],
        "summary":            final_state["rca_summary"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rca_agent.py "<natural language query>"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nQuery : {query}\n{'='*60}")

    result = run_rca_workflow(query)

    print(f"Site             : {result['site_name']}")
    print(f"Date             : {result['log_date']}")
    print(f"Root Cause       : {result['dominant_rca_label']}")
    print(f"Confidence Score : {result['confidence_score']:.2f} / 1.00")
    print(f"\nRCA Summary\n{'-'*60}")
    print(result["summary"])
    print("=" * 60)
