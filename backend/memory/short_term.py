"""
short_term.py - Short-Term Memory Utilities
=============================================

PURPOSE:
    Helper functions for managing short-term memory within
    a single graph session.

SHORT-TERM MEMORY = GraphState itself.
    The graph state IS the short-term memory.
    It holds everything from the current conversation:
    - The query
    - Tool results
    - The draft
    - The trace

    This module provides utilities for initializing state
    and formatting messages for the LLM.

WHY A SEPARATE FILE?
    The actual state IS in GraphState (graph/state.py).
    This file provides helper functions to work with it.
    Keeping helpers separate from the schema definition
    makes both files cleaner.
"""

from datetime import datetime
from uuid import uuid4
from graph.state import GraphState


def create_initial_state(query: str, session_id: str = None) -> dict:
    """
    Creates the initial graph state for a new query.

    Called when a new /query request comes in.
    Sets required fields and None for everything else.

    WHY NOT JUST PASS {query, session_id}?
        LangGraph requires ALL TypedDict fields to be present
        (even if None) when using interrupt_before.
        Without this, it raises a KeyError when accessing
        optional fields that haven't been set yet.

    session_id:
        If provided, uses it (for continuing a session).
        If None, generates a new UUID.
        UUID = Universally Unique Identifier - a 128-bit random
        number formatted as "550e8400-e29b-41d4-a716-446655440000".
        Guaranteed unique across all machines worldwide.
    """

    if session_id is None:
        session_id = str(uuid4())

    return {
        # Input
        "query": query,
        "session_id": session_id,

        # Planner (not set yet)
        "plan": None,
        "plan_reasoning": None,

        # Tool results (not set yet)
        "rag_results": None,
        "search_results": None,
        "reddit_results": None,
        "tools_used": None,

        # Draft (not set yet)
        "draft_answer": None,

        # Verification (not set yet)
        "verification_passed": None,
        "verification_feedback": None,
        "retry_count": 0,

        # Approval (not set yet)
        "human_approved": None,
        "human_feedback": None,

        # Output (not set yet)
        "final_answer": None,

        # Memory
        "long_term_context": None,
        "messages": [
            {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat(),
            }
        ],

        # Observability
        "trace": [f"Session started: {session_id[:8]}... | Query received"],
        "error": None,
    }


def format_state_for_display(state: dict) -> dict:
    """
    Formats graph state for sending to the frontend.

    The full state has many internal fields the frontend doesn't need.
    This function extracts just what the UI displays:
    - trace (for the tool call log)
    - draft_answer (for the review panel)
    - final_answer (for the result panel)
    - status (what phase the graph is in)
    - session_id (for the /approve call)

    STATUS VALUES:
        "planning"   - planner node is running
        "executing"  - tools are being called
        "drafting"   - writing the answer
        "verifying"  - checking quality
        "awaiting_approval" - waiting for human
        "finalizing" - writing final answer
        "complete"   - done
        "error"      - something went wrong
    """

    # Determine current status based on what fields are set
    if state.get("final_answer"):
        status = "complete"
    elif state.get("human_approved"):
        status = "finalizing"
    elif state.get("draft_answer") and not state.get("verification_passed"):
        status = "verifying"
    elif state.get("draft_answer"):
        status = "awaiting_approval"
    elif state.get("rag_results") or state.get("search_results"):
        status = "drafting"
    elif state.get("plan"):
        status = "executing"
    elif state.get("error"):
        status = "error"
    else:
        status = "planning"

    return {
        "session_id": state.get("session_id"),
        "status": status,
        "query": state.get("query"),
        "trace": state.get("trace", []),
        "draft_answer": state.get("draft_answer"),
        "final_answer": state.get("final_answer"),
        "tools_used": state.get("tools_used", []),
        "verification_passed": state.get("verification_passed"),
        "error": state.get("error"),
    }