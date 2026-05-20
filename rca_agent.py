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

Multi-turn support:
  The compiled graph uses a MemorySaver checkpointer keyed by thread_id.
  Each conversation thread retains full message history across calls, so
  the parse_query node can resolve relative follow-up questions ("the next
  day", "cell 0 instead") by inspecting prior turns in the message log.

Usage:
    python rca_agent.py "What is the situation of Nashik site on 1 Jan 2026?"
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.pipelines.inference_pipeline import InferencePipeline
from logger.logger import logging

from utils.utils import save_conversations

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
# Single-entry cache: stores the last fetched DataFrame keyed by (site, date, cell, ue).
# Eliminates the duplicate Databricks SQL query when run_inference and fetch_data
# are called back-to-back for the same arguments within one LangGraph invocation.
_fetch_cache: dict = {}


def _get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


def _cached_fetch(site_name: str, log_date: str, cellid, ueid) -> "pd.DataFrame | None":
    key = (site_name.lower(), log_date, cellid, ueid)
    if key not in _fetch_cache:
        _fetch_cache.clear()  # keep memory bounded to one entry
        _fetch_cache[key] = _get_pipeline().fetch_data(
            log_date=log_date,
            site_name=site_name.lower(),
            cellid=cellid,
            ueid=ueid,
        )
    return _fetch_cache[key]


# ---------------------------------------------------------------------------
# LangChain tools (decorated with @tool so LangGraph ToolNode can run them)
# ---------------------------------------------------------------------------

@tool
def run_inference(
    site_name: str,
    log_date: str,
    cellid: int,
    ueid: int,
) -> str:
    """Run the telecom RCA LSTM inference model for a given site and date.

    Returns predicted RCA category and estimated issue start time per UE session.
    Always call this tool first before fetch_data.

    Args:
        site_name: Telecom site name, e.g. 'nashik'.
        log_date:  Date in YYYY-MM-DD format.
        cellid:    cell ID to filter results.
        ueid:      UE ID to filter results.
    """
    logging.info(f"[Tool] run_inference site={site_name} date={log_date} cell={cellid} ue={ueid}")
    raw_df = _cached_fetch(site_name, log_date, cellid, ueid)
    df: pd.DataFrame | None = _get_pipeline().predict(raw_df) if raw_df is not None and not raw_df.empty else None
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
    cellid: int,
    ueid: int,
) -> str:
    """Fetch KPI time-series data (CQI, MCS, BLER metrics) aggregated into 30-minute windows.

    Returns per-session windowed KPI means so you can identify when signal quality
    degraded over the course of the session. Use this to verify inference results
    against the observed KPI trends.
    Always call run_inference before calling this tool.

    Args:
        site_name: Telecom site name, e.g. 'nashik'.
        log_date:  Date in YYYY-MM-DD format.
        cellid:    cell ID to filter results.
        ueid:      UE ID to filter results.
    """
    logging.info(f"[Tool] fetch_data site={site_name} date={log_date} cell={cellid} ue={ueid}")
    df: pd.DataFrame | None = _cached_fetch(site_name, log_date, cellid, ueid)
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
   - High BLER values (ibler/rbler/resbler/tbler > 10) indicate link errors.
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
# Query parser node — multi-turn aware
# ---------------------------------------------------------------------------

PARSER_SYSTEM = """Extract telecom query parameters from a natural-language question.

You will receive the full conversation history. Use prior turns to resolve any
relative or implicit references in the latest question.
Rules:
- If the latest question omits a parameter (site, date, cell, UE), inherit it
  from the most recent prior turn that specified it.
- Resolve relative dates: "the next day" means +1 day from the prior date,
  "yesterday" means -1 day, etc.
- Explicit values in the latest question always override inherited ones.
- If there is a follow up question, you need to analyze the prior context
    and identify the next UE IDs in sequence to analyze in the current turn along with all the past UE IDs.

Return ONLY a JSON object with these keys:
  site_name (str, lowercase),
  log_date   (str, YYYY-MM-DD),
  cellid     (int),
  ueids      (list of int) — list of explicit UE IDs to analyse;
"""
# Examples:
##   Single UE: "What is the situation of Nashik site on 5 Jan 2026 for cell 1 ueid 17027?"
##   -> {"site_name": "nashik", "log_date": "2026-01-05", "cellid": 1, "ueids": [17027]}

##   Follow-up (after the above): "Can you check cell 0 the next day?"
##   -> Add the next 2 UE IDs in sequence to the last UE ID from the prior context
  
##   Multi-UE comparison: "Compare UE 17019, 17020, 17021 on Bangalore cell 0 on 5 Jan 2026"
##   -> {"site_name": "bangalore", "log_date": "2026-01-05", "cellid": 0, "ueids": [17019, 17020, 17021]}

##   Follow-up (after the above): "Can you compare it with the next 2 UEs?"
##   -> Add the next 2 UE IDs in sequence to the last UE ID from the prior context

##   Auto-discover sequential UEs: "For SIT site on 9 Jan 2026, can you compare the ue 17012 to 17018 for cell 1?"
##   -> {"site_name": "sit", "log_date": "2026-01-09", "cellid": 1, "ueids": [17012, 17013, 17014, 17015, 17016, 17017, 17018]}

##   Follow-up (after the above): "Can you compare it with the next 2 UEs?"
##   -> Add the next 2 UE IDs in sequence to the last UE ID from the prior context

def _build_parser_context(messages: list) -> str:
    """Render prior human/AI turns into a compact context string for the parser."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            # Only include the text content (not tool call blobs) to keep it concise.
            # lines.append(f"Assistant summary: {str(m.content)[:300]}")
            lines.append(f"Assistant summary: {str(m.content)}")
    return "\n".join(lines)


def parse_query(natural_query: str, prior_messages: list | None = None) -> dict:
    """Extract structured query params from a natural-language query.

    Returns a dict with keys: site_names (list), log_date, cellid, ueids (list).
    """
    # prior_context = _build_parser_context(prior_messages) if prior_messages else ""

    # user_content = natural_query
    # if prior_context:
    #     user_content = (
    #         f"Conversation so far:\n\n{prior_context}\n\n"
    #         f"Latest question to parse:\n\n{natural_query}"
    #     )

    # logging.info(f"[parse_query] Prior Context:\n{user_content}")

    messages = [SystemMessage(content=PARSER_SYSTEM)]
    messages += prior_messages if prior_messages else []
    messages += [HumanMessage(content=natural_query)]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    try:
        params = json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        params = json.loads(match.group()) if match else {}

    # Normalise ueids: accept ueids list, or legacy scalar ueid
    ueids: list[int] = params.get("ueids") or []
    # if not ueids and params.get("ueid") is not None:
    #     ueids = [int(params["ueid"])]

    return {
        "site_name": params.get("site_name", ""),
        "log_date":   params.get("log_date", ""),
        "cellid":     int(params.get("cellid")),
        "ueids":      ueids,
    }


def parse_query_node(state: RCAState) -> dict:
    """LangGraph node: extract structured query params from the latest HumanMessage."""
    messages = state["messages"]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    latest_query = human_messages[-1].content if human_messages else ""
    prior_messages = messages[:-1] if len(messages) > 1 else []

    params = parse_query(latest_query, prior_messages)
    ueids = params["ueids"]

    return {
        "site_name": params["site_name"],
        "log_date":  params["log_date"],
        "cellid":    params["cellid"],
        "ueid":      ueids[0] if ueids else None,
    }


# ---------------------------------------------------------------------------
# Build the LangGraph — compiled once at module load, reused across requests
# ---------------------------------------------------------------------------

# In-process checkpointer. Each unique thread_id gets its own isolated
# conversation history. To persist across process restarts (e.g. multi-worker
# deployments), swap MemorySaver for langgraph-checkpoint-sqlite's SqliteSaver
# or a Redis-backed checkpointer.
_CHECKPOINTER = MemorySaver()


def _build_graph():
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

    return graph.compile(checkpointer=_CHECKPOINTER)


_COMPILED_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rca_workflow(natural_query: str, thread_id: str | None = None, save_history: bool = True) -> dict[str, Any]:
    """
    Run the full agentic RCA workflow from a natural-language query.

    Args:
        natural_query: e.g. "What is the situation of Nashik site on 1 Jan 2026?"
        thread_id:     Conversation thread ID for multi-turn continuity. If None,
                       a new thread is created and its ID is returned in the result.
                       Pass the same thread_id on follow-up questions to inherit
                       prior context (site, date, cell, UE) and resolve relative
                       references like "the next day" or "cell 0 instead".

    Returns:
        dict with keys: thread_id, site_name, log_date, dominant_rca_label,
                        confidence_score, summary
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    app = _COMPILED_GRAPH
    config = {"configurable": {"thread_id": thread_id}}

    # With a checkpointer, we only pass the new message; LangGraph merges it
    # with the stored history for this thread automatically.
    logging.info(f"[run_rca_workflow] Natural query:\n{natural_query}\n")
    input_state = {"messages": [HumanMessage(content=natural_query)]}

    final_state = app.invoke(input_state, config=config)
    if save_history:
        save_conversations(
            thread_id=thread_id,
            messages=[HumanMessage(content=natural_query)] + [AIMessage(content=final_state["rca_summary"])]
        )

    return {
        "thread_id":          thread_id,
        "site_name":          final_state["site_name"],
        "log_date":           final_state["log_date"],
        "dominant_rca_label": final_state["judge_json"].get("dominant_rca_label", "Unknown"),
        "confidence_score":   final_state["confidence_score"],
        "summary":            final_state["rca_summary"],
    }


# ---------------------------------------------------------------------------
# Multi-UE comparison
# ---------------------------------------------------------------------------

COMPARISON_SUMMARIZER_SYSTEM = """You are a senior telecom network operations engineer.
You have received RCA reports for multiple UEs on the same site and must write a
concise cross-UE comparison for the network operations team.

Required format:
1. One sentence: the site, cell, date, and the overall picture across UEs.
2. Per-UE breakdown — for each UE: dominant RCA label, confidence score, and the
   single most important KPI anomaly (or "no issues detected").
3. Cross-UE pattern — what do the UEs have in common, and where do they diverge?
   Are issues isolated to specific UEs or widespread across the cell?
4. Recommended actions — per UE if they differ, or a single recommendation if uniform.

Keep the report under 350 words. Plain English only — no JSON, no code blocks."""


def run_rca_ue_comparison(
    params: dict,
    natural_query: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run parallel per-UE RCA workflows and return a combined comparison result.

    When ueids are specified in the query they are used directly; otherwise all
    distinct UE IDs for the site/cell/date are fetched from Databricks first.

    Returns:
        dict with keys: thread_id, sites (list of per-UE dicts reusing SiteResult
                        schema), log_date, comparison_summary, is_comparison (True)
    """
    from utils.utils import fetch_distinct_ues

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # params   = parse_query(natural_query)
    site_name = params["site_name"]
    log_date  = params["log_date"]
    cellid    = params["cellid"]
    ueids     = params["ueids"]

    # if not ueids:
    #     logging.info(f"[run_rca_ue_comparison] Auto-discovering UEs for {site_name}/{cellid}/{log_date}")
    #     ueids = fetch_distinct_ues(log_date, site_name, cellid)
    #     if not ueids:
    #         return {
    #             "thread_id":          thread_id,
    #             "is_comparison":      True,
    #             "sites":              [],
    #             "log_date":           log_date,
    #             "comparison_summary": "No UE data found for the requested site/cell/date.",
    #         }

    per_ue_results: dict[int, dict] = {}

    def _run_one(site_name: str, log_date: str, cellid: int, ue: int) -> tuple[int, dict]:
        parts = [f"Analyse telecom site '{site_name}' for {log_date}"]
        parts.append(f"cell {cellid}")
        parts.append(f"UE {ue}")
        single_query = " ".join(parts) + "."
        result = run_rca_workflow(single_query, thread_id=None, save_history=False)
        return ue, result

    with ThreadPoolExecutor(max_workers=len(ueids)) as pool:
        futures = [pool.submit(_run_one, site_name, log_date, cellid, ue) for ue in ueids]
        for future in as_completed(futures):
            ue, result = future.result()
            per_ue_results[ue] = result

    ue_blocks = []
    for ue in ueids:
        r = per_ue_results.get(ue, {})
        ue_blocks.append(
            f"UE {ue}:\n"
            f"  RCA: {r.get('dominant_rca_label', 'Unknown')}\n"
            f"  Confidence: {r.get('confidence_score', 0.0):.2f}\n"
            f"  Summary: {r.get('summary', 'No data')}"
        )
    context = (
        f"Site: {site_name}\n"
        f"Date: {log_date}\n"
        + (f"Cell: {cellid}\n" if cellid is not None else "")
        + "\n\n"
        + "\n\n".join(ue_blocks)
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    comparison_response = llm.invoke([
        SystemMessage(content=COMPARISON_SUMMARIZER_SYSTEM),
        HumanMessage(content=context),
    ])

    save_conversations(
        thread_id=thread_id,
        messages=[HumanMessage(content=natural_query)] + [AIMessage(content=comparison_response.content.strip())]
    )

    ues_out = [
        {
            "site_name":          f"UE {ue}",
            "dominant_rca_label": per_ue_results.get(ue, {}).get("dominant_rca_label", "Unknown"),
            "confidence_score":   per_ue_results.get(ue, {}).get("confidence_score", 0.0),
            "summary":            per_ue_results.get(ue, {}).get("summary", ""),
        }
        for ue in ueids
    ]

    return {
        "thread_id":          thread_id,
        "is_comparison":      True,
        "sites":              ues_out,
        "log_date":           log_date,
        "comparison_summary": comparison_response.content.strip(),
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

    print(f"Thread ID        : {result['thread_id']}")
    print(f"Site             : {result['site_name']}")
    print(f"Date             : {result['log_date']}")
    print(f"Root Cause       : {result['dominant_rca_label']}")
    print(f"Confidence Score : {result['confidence_score']:.2f} / 1.00")
    print(f"\nRCA Summary\n{'-'*60}")
    print(result["summary"])
    print("=" * 60)
