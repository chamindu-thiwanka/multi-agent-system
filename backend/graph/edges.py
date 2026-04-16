"""
edges.py - Graph Routing Logic
================================

PURPOSE:
    Defines the CONDITIONAL EDGES in our LangGraph state machine.
    Edges decide WHERE to go next based on the current state.

WHAT IS AN EDGE?
    In a graph:
    - NODES = things that happen (planner, executor, verifier, etc.)
    - EDGES = connections between nodes (what leads to what)

    UNCONDITIONAL EDGE: always goes from A to B
        graph.add_edge("planner", "tool_executor")
        "after planner, ALWAYS go to tool_executor"

    CONDITIONAL EDGE: goes to different places based on state
        graph.add_conditional_edges(
            "verifier",
            should_retry,         ← this function decides
            {
                "retry": "planner",   ← if returns "retry"
                "continue": "human_checkpoint"  ← if returns "continue"
            }
        )

WHY CONDITIONAL EDGES?
    The whole point of LangGraph over a simple sequential script
    is the ability to BRANCH and LOOP based on runtime conditions.

    The verifier-to-planner retry loop is ONLY possible because
    of conditional edges. A linear script can't loop back.

FUNCTIONS IN THIS FILE:
    route_after_verifier  - should we retry or continue to human?
    route_after_checkpoint - should we finalize or wait?
"""

import logging
from graph.state import GraphState

logger = logging.getLogger(__name__)


def route_after_verifier(state: GraphState) -> str:
    """
    Routing function called after the verifier node runs.

    RETURNS one of these string keys (must match keys in add_conditional_edges):
        "retry"    - verification failed, go back to planner
        "continue" - verification passed, go to human checkpoint

    HOW LANGGRAPH USES THIS:
        graph.add_conditional_edges(
            "verifier_node",
            route_after_verifier,      ← this function
            {
                "retry": "planner_node",
                "continue": "human_checkpoint_node"
            }
        )

        After verifier runs:
        1. LangGraph calls route_after_verifier(current_state)
        2. Function returns "retry" or "continue"
        3. LangGraph routes to "planner_node" or "human_checkpoint_node"
    """

    verification_passed = state.get("verification_passed", True)
    retry_count = state.get("retry_count", 0)

    if verification_passed:
        logger.info("[Router] Verification passed -> human checkpoint")
        return "continue"
    elif retry_count >= 2:
        # Force continue after too many retries
        # We don't want infinite loops
        logger.info("[Router] Max retries reached -> forcing human checkpoint")
        return "continue"
    else:
        logger.info(
            f"[Router] Verification failed (attempt {retry_count}) "
            f"-> retrying from planner"
        )
        return "retry"


def route_after_checkpoint(state: GraphState) -> str:
    """
    Routing function called after the human checkpoint node runs.

    NOTE: Due to how LangGraph interrupt_before works, this function
    runs AFTER the human has approved (the node itself only runs
    post-approval). So this should almost always return "finalize".

    We include this check as a safety measure.

    RETURNS:
        "finalize" - approved, go to final output
        "wait"     - not approved (shouldn't happen, but safe fallback)
    """

    human_approved = state.get("human_approved", False)

    if human_approved:
        logger.info("[Router] Human approved -> final output")
        return "finalize"
    else:
        # This shouldn't happen in normal flow
        # But if it does, we finalize anyway to avoid getting stuck
        logger.warning("[Router] Checkpoint reached without approval, finalizing anyway")
        return "finalize"