"""
graph.py - LangGraph State Machine Assembly
=============================================

PURPOSE:
    Assembles all nodes and edges into a compiled LangGraph.
    This is the "blueprint" of our entire agent pipeline.

HOW LANGGRAPH WORKS:
    1. Define a StateGraph with your state type
    2. Add nodes (functions that process state)
    3. Add edges (connections between nodes)
    4. Set entry point (where execution starts)
    5. Compile (validates the graph, prepares for execution)

    The compiled graph is like a program you can run:
        result = graph.invoke({"query": "...", "session_id": "..."})

THE INTERRUPT MECHANISM (Human-in-the-Loop):
    graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_checkpoint_node"]
    )

    interrupt_before=["human_checkpoint_node"] tells LangGraph:
    "Before executing human_checkpoint_node, STOP."

    When the graph reaches human_checkpoint_node:
    1. LangGraph saves entire state to checkpointer (SQLite)
    2. Graph execution pauses
    3. invoke() returns with status "interrupted"
    4. FastAPI sends draft to frontend
    5. Human reviews and calls POST /approve
    6. FastAPI calls graph.invoke() again with same thread_id
    7. LangGraph loads state from checkpointer
    8. human_checkpoint_node finally executes
    9. Graph continues to final_output_node

THE CHECKPOINTER (State Persistence):
    SqliteSaver saves graph state to a SQLite database.
    Each "run" is identified by a "thread_id" (our session_id).
    This allows:
    - Pausing mid-execution and resuming later
    - Multiple concurrent users (different thread_ids)
    - Crash recovery (state is persisted to disk)

COMPLETE GRAPH FLOW:
    START
      |
      v
    planner_node
      |
      v
    tool_executor_node
      |
      v
    draft_node
      |
      v
    verifier_node
      |
      +--[retry]---> planner_node (loop back)
      |
      +--[continue]->
      |
      v
    [PAUSE - wait for human approval]
      |
      v
    human_checkpoint_node
      |
      v
    final_output_node
      |
      v
    END
"""

import logging
import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import GraphState
from graph.nodes import (
    planner_node,
    tool_executor_node,
    draft_node,
    verifier_node,
    human_checkpoint_node,
    final_output_node,
)
from graph.edges import route_after_verifier, route_after_checkpoint
from config import get_settings

logger = logging.getLogger(__name__)


def create_graph():
    """
    Creates, configures, and compiles the LangGraph state machine.

    Returns the compiled graph ready for invocation.

    WHY A FUNCTION (not module-level code)?
        We create the graph inside a function so:
        1. It's only created when actually needed
        2. The checkpointer path comes from settings (configurable)
        3. Tests can create fresh graphs without global state
        4. Multiple graph instances are possible

    THE CHECKPOINTER:
        SqliteSaver.from_conn_string() creates a SQLite database
        at the specified path. This database stores:
        - Complete graph state snapshots
        - One snapshot per "step" in the graph
        - Identified by thread_id (our session_id)

        When graph resumes after human approval:
        - LangGraph reads latest snapshot for that thread_id
        - Restores complete state (all tool results, draft, etc.)
        - Continues from the interrupted point
    """

    settings = get_settings()

    # Path to SQLite database for storing graph checkpoints
    # This is the GRAPH STATE checkpoint (different from long-term memory)
    checkpoint_db_path = str(
        Path(__file__).parent.parent.parent / "graph_checkpoints.db"
    )

    logger.info(f"Creating graph with checkpoint db: {checkpoint_db_path}")

    # # Create the checkpointer
    # # SqliteSaver persists graph state to SQLite
    # # This enables the human-in-the-loop pause/resume feature
    # checkpointer = SqliteSaver.from_conn_string(checkpoint_db_path)

    conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ============================================================
    # CREATE STATE GRAPH
    # ============================================================

    # StateGraph(GraphState) means this graph uses GraphState
    # as its state schema. LangGraph validates all state updates
    # against this schema.
    workflow = StateGraph(GraphState)

    # ============================================================
    # ADD NODES
    # ============================================================
    # Each node is a Python function from nodes.py
    # The string name is how we reference the node in edges

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("tool_executor_node", tool_executor_node)
    workflow.add_node("draft_node", draft_node)
    workflow.add_node("verifier_node", verifier_node)
    workflow.add_node("human_checkpoint_node", human_checkpoint_node)
    workflow.add_node("final_output_node", final_output_node)

    # ============================================================
    # SET ENTRY POINT
    # ============================================================
    # Which node runs first when graph.invoke() is called?
    workflow.set_entry_point("planner_node")

    # ============================================================
    # ADD UNCONDITIONAL EDGES
    # ============================================================
    # These always go from A to B, no conditions.

    # After planning, always execute the tools
    workflow.add_edge("planner_node", "tool_executor_node")

    # After executing tools, always generate a draft
    workflow.add_edge("tool_executor_node", "draft_node")

    # After draft, always verify quality
    workflow.add_edge("draft_node", "verifier_node")

    # After human approval, always generate final output
    workflow.add_edge("human_checkpoint_node", "final_output_node")

    # After final output, end the graph
    workflow.add_edge("final_output_node", END)

    # ============================================================
    # ADD CONDITIONAL EDGES
    # ============================================================
    # These route to different nodes based on state values.

    # After verifier: either retry (back to planner) or continue
    workflow.add_conditional_edges(
        "verifier_node",           # source node
        route_after_verifier,      # routing function
        {
            "retry": "planner_node",              # if returns "retry"
            "continue": "human_checkpoint_node",  # if returns "continue"
        }
    )

    # After checkpoint: route to final output
    # (This is mainly for clarity - always goes to final_output)
    workflow.add_conditional_edges(
        "human_checkpoint_node",
        route_after_checkpoint,
        {
            "finalize": "final_output_node",
            "wait": "final_output_node",  # fallback also goes to final
        }
    )

    # ============================================================
    # COMPILE GRAPH
    # ============================================================
    # compile() validates the graph structure and prepares it
    # for execution. It checks:
    # - All nodes referenced in edges actually exist
    # - Entry point is defined
    # - All conditional edge return values have corresponding routes
    # - No unreachable nodes

    compiled_graph = workflow.compile(
        checkpointer=checkpointer,

        # interrupt_before: pause execution BEFORE these nodes
        # The graph will stop before human_checkpoint_node runs
        # and wait for resume signal
        interrupt_before=["human_checkpoint_node"],
    )

    logger.info("Graph compiled successfully")
    return compiled_graph


# ================================================================
# GRAPH SINGLETON
# ================================================================
# We create the graph once and reuse it.
# Creating it multiple times would create multiple SQLite connections.

_graph_instance = None


def get_graph():
    """
    Returns the singleton graph instance.
    Creates it on first call, reuses on subsequent calls.
    """
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_graph()
    return _graph_instance