"""
routes.py - FastAPI Route Definitions
=======================================

PURPOSE:
    Defines the three core API endpoints required by the assignment:
    POST /query   - starts the agent pipeline
    POST /approve - resumes after human approval  
    GET  /trace   - returns execution trace

PLUS additional helpful endpoints:
    GET  /status  - check current state of a session
    GET  /sessions - list all sessions (for debugging)
    POST /stream  - streaming version of /query using SSE

WHY SEPARATE routes.py FROM main.py?
    main.py handles application-level concerns:
    - Creating the app
    - Adding middleware
    - Startup/shutdown
    
    routes.py handles business logic:
    - What happens when /query is called
    - What happens when /approve is called
    
    Separating them means:
    - Easier to test routes independently
    - main.py stays clean and focused
    - Routes can be split further as app grows
    - This is the standard FastAPI pattern

PYDANTIC REQUEST/RESPONSE MODELS:
    We define Pydantic models for all request bodies and responses.
    
    WHY?
    1. Automatic validation: if required field missing → 422 error with helpful message
    2. Type conversion: if "5" sent for int field → automatically converts
    3. Documentation: FastAPI reads these models to generate /docs UI
    4. Autocomplete: IDE knows the shape of request/response objects

HOW LANGGRAPH THREADING WORKS WITH FASTAPI:
    LangGraph's pipeline is synchronous (blocking).
    FastAPI is asynchronous (non-blocking).
    
    We bridge them using run_in_executor():
        await loop.run_in_executor(None, sync_function)
    
    This runs the sync function in a thread pool,
    so FastAPI can handle other requests while the LLM thinks.
    Without this, one LLM call would freeze the entire server.
"""

import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph import get_graph
from memory.short_term import create_initial_state, format_state_for_display
from memory.long_term import load_relevant_context

logger = logging.getLogger(__name__)

# Create router - all routes in this file are prefixed with /api/v1
# (the prefix is added in main.py when we do app.include_router)
router = APIRouter()


# ================================================================
# REQUEST / RESPONSE MODELS
# ================================================================

class QueryRequest(BaseModel):
    """
    Request body for POST /query.
    
    Pydantic validates this automatically.
    If 'query' is missing from request body, FastAPI returns:
    {"detail": [{"loc": ["body", "query"], "msg": "field required"}]}
    """
    
    query: str = Field(
        ...,  # ... means required (no default)
        description="The user's research question",
        min_length=3,
        max_length=1000,
        example="What are recent AI approaches for detecting Alzheimer's disease?"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID for continuing a conversation. "
            "If not provided, a new session is created."
        ),
        example="550e8400-e29b-41d4-a716-446655440000"
    )


class ApproveRequest(BaseModel):
    """Request body for POST /approve."""
    
    session_id: str = Field(
        ...,
        description="Session ID from the /query response",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    
    approved: bool = Field(
        default=True,
        description="True to approve the draft, False to reject"
    )
    
    feedback: Optional[str] = Field(
        default=None,
        description="Optional feedback if rejecting the draft",
        example="The answer needs more recent citations"
    )


class QueryResponse(BaseModel):
    """Response from POST /query."""
    
    session_id: str
    status: str  # "awaiting_approval", "error", etc.
    query: str
    draft_answer: Optional[str] = None
    trace: list = []
    tools_used: list = []
    error: Optional[str] = None


class ApproveResponse(BaseModel):
    """Response from POST /approve."""
    
    session_id: str
    status: str  # "complete", "error"
    final_answer: Optional[str] = None
    trace: list = []
    error: Optional[str] = None


class TraceResponse(BaseModel):
    """Response from GET /trace."""
    
    session_id: str
    trace: list = []
    status: str
    tools_used: list = []


# ================================================================
# HELPER: RUN GRAPH SYNCHRONOUSLY IN THREAD
# ================================================================

def _run_graph_sync(state: dict, config: dict) -> dict:
    """
    Runs the graph synchronously.
    
    This function is called in a thread pool by run_in_executor.
    It blocks while running, which is fine in a thread
    (doesn't block the async event loop).
    
    Collects all state from graph streaming until interrupted.
    Returns the final accumulated state.
    """
    graph = get_graph()
    
    # Collect all state updates from the graph stream
    # graph.stream() yields {node_name: state_update} for each node
    final_state = {}
    
    for event in graph.stream(state, config):
        # Each event is {node_name: state_updates}
        for node_name, node_state in event.items():
            if node_name == "__interrupt__":
                # Graph hit the interrupt point (human checkpoint)
                # This is expected - not an error
                logger.info("Graph interrupted at human checkpoint")
                break
            # Merge node state into our accumulated state
            final_state.update(node_state)
    
    return final_state


def _resume_graph_sync(config: dict) -> dict:
    """
    Resumes the graph after human approval.
    
    When called with the same config (thread_id) after update_state(),
    LangGraph loads the saved checkpoint and continues from where
    it stopped (after the human_checkpoint_node).
    
    Passing None as state tells LangGraph "resume from checkpoint".
    """
    graph = get_graph()
    
    final_state = {}
    
    for event in graph.stream(None, config):
        for node_name, node_state in event.items():
            if node_name != "__interrupt__":
                final_state.update(node_state)
    
    return final_state


# ================================================================
# ENDPOINT 1: POST /query
# ================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a research query",
    description=(
        "Starts the multi-agent pipeline for a research query. "
        "The pipeline runs: Planner → Tools → Draft → Verifier, "
        "then pauses for human approval. "
        "Returns the draft answer and execution trace."
    ),
)
async def submit_query(request: QueryRequest):
    """
    Starts the multi-agent pipeline.
    
    FLOW:
    1. Load long-term memory context for this session
    2. Create initial graph state
    3. Run graph in thread pool (non-blocking)
    4. Graph pauses at human checkpoint
    5. Return draft and trace to frontend
    
    The session_id from the response must be used in /approve.
    
    WHAT "async" MEANS HERE:
    The 'async def' means FastAPI handles this endpoint
    asynchronously. When we 'await' something (like run_in_executor),
    FastAPI can handle other requests while waiting.
    Without async, one slow LLM call would freeze the entire server.
    """
    
    logger.info(f"[API] POST /query: '{request.query[:80]}...'")
    
    try:
        # Create initial state with all required fields
        state = create_initial_state(
            query=request.query,
            session_id=request.session_id,
        )
        session_id = state["session_id"]
        
        # Load long-term context from previous conversations
        # This gives the agent memory of past sessions
        long_term_context = load_relevant_context(
            session_id=session_id,
            query=request.query,
        )
        if long_term_context:
            state["long_term_context"] = long_term_context
            logger.info(f"[API] Loaded long-term context for session {session_id[:8]}")
        
        # LangGraph config - thread_id identifies this run
        # All graph operations for this session use the same thread_id
        config = {"configurable": {"thread_id": session_id}}
        
        # Run graph in a thread pool
        # run_in_executor(None, fn, *args) runs fn(*args) in a thread
        # 'None' means use the default ThreadPoolExecutor
        # 'await' means "wait for it, but don't block other requests"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _run_graph_sync,
            state,
            config,
        )
        
        # After graph pauses, get the current state from checkpointer
        graph = get_graph()
        current_state = graph.get_state(config)
        state_dict = dict(current_state.values)
        
        # Format for frontend consumption
        display = format_state_for_display(state_dict)
        
        logger.info(
            f"[API] Query complete: session={session_id[:8]} "
            f"status={display['status']}"
        )
        
        return QueryResponse(
            session_id=session_id,
            status=display["status"],
            query=request.query,
            draft_answer=display.get("draft_answer"),
            trace=display.get("trace", []),
            tools_used=display.get("tools_used", []),
            error=display.get("error"),
        )
        
    except Exception as e:
        logger.error(f"[API] Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )


# ================================================================
# ENDPOINT 2: POST /approve
# ================================================================

@router.post(
    "/approve",
    response_model=ApproveResponse,
    summary="Approve or reject the draft answer",
    description=(
        "Resumes the pipeline after human review. "
        "If approved, generates the final formatted answer. "
        "session_id must match the one from /query."
    ),
)
async def approve_draft(request: ApproveRequest):
    """
    Resumes the paused graph after human approval.
    
    FLOW:
    1. Load the paused graph state from checkpointer
    2. Update state with human_approved = True
    3. Resume graph execution
    4. Graph runs: human_checkpoint_node → final_output_node
    5. Return final answer
    
    WHY TWO STEPS (update_state then stream)?
        update_state() injects our approval into the saved state.
        stream(None, config) resumes from the saved state.
        
        If we just called stream with new state, it would START OVER
        from the beginning, losing all the tool results and draft.
        By using the same thread_id and passing None, LangGraph
        CONTINUES from exactly where it stopped.
    """
    
    logger.info(
        f"[API] POST /approve: session={request.session_id[:8]} "
        f"approved={request.approved}"
    )
    
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Verify this session exists
        current_state = graph.get_state(config)
        if current_state.values is None or not dict(current_state.values):
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found. "
                       f"Submit a query first."
            )
        
        if not request.approved:
            # Rejection: for now, we still finalize
            # Full rejection flow (retry with feedback) would require
            # resetting retry_count and adding rejection feedback
            # This is a future enhancement
            logger.info(
                f"[API] Draft rejected by human. "
                f"Feedback: {request.feedback}"
            )
        
        # Inject human approval into the saved graph state
        # update_state() modifies the checkpoint for this thread_id
        # The next stream() call will see human_approved = True
        graph.update_state(
            config,
            {
                "human_approved": request.approved,
                "human_feedback": request.feedback or "",
            }
        )
        
        # Resume graph execution in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _resume_graph_sync,
            config,
        )
        
        # Get final state
        final_state = graph.get_state(config)
        state_dict = dict(final_state.values)
        display = format_state_for_display(state_dict)
        
        logger.info(
            f"[API] Approval complete: session={request.session_id[:8]} "
            f"status={display['status']}"
        )
        
        return ApproveResponse(
            session_id=request.session_id,
            status=display["status"],
            final_answer=display.get("final_answer"),
            trace=display.get("trace", []),
            error=display.get("error"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Approval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Approval failed: {str(e)}"
        )


# ================================================================
# ENDPOINT 3: GET /trace
# ================================================================

@router.get(
    "/trace",
    response_model=TraceResponse,
    summary="Get execution trace for a session",
    description=(
        "Returns the full execution trace showing which tools were called, "
        "what decisions were made, and the pipeline status. "
        "Use this to show users how their answer was generated."
    ),
)
async def get_trace(session_id: str):
    """
    Returns the execution trace for a session.
    
    The trace is an ordered list of strings showing what happened:
    ["Planner: selected tools [rag, internet_search]",
     "RAG Tool: retrieved 5 articles from PubMed",
     "Search Tool: found 3 results via Tavily",
     ...]
    
    This is what the frontend displays in the "How it worked" panel.
    
    Query parameter: GET /api/v1/trace?session_id=550e8400-...
    """
    
    logger.info(f"[API] GET /trace: session={session_id[:8]}")
    
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": session_id}}
        
        state = graph.get_state(config)
        
        if state.values is None or not dict(state.values):
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        state_dict = dict(state.values)
        display = format_state_for_display(state_dict)
        
        return TraceResponse(
            session_id=session_id,
            trace=display.get("trace", []),
            status=display["status"],
            tools_used=display.get("tools_used", []),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Trace failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trace: {str(e)}"
        )


# ================================================================
# ENDPOINT 4: GET /status (bonus - helpful for frontend polling)
# ================================================================

@router.get(
    "/status/{session_id}",
    summary="Get current status of a session",
    description="Returns the current status and all available data for a session.",
)
async def get_status(session_id: str):
    """
    Returns full current state of a session.
    
    Frontend can poll this to check if processing is complete.
    More detailed than /trace - includes draft, final answer, etc.
    
    Path parameter: GET /api/v1/status/550e8400-...
    (Note: path parameter, not query parameter - it's in the URL)
    """
    
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": session_id}}
        
        state = graph.get_state(config)
        
        if state.values is None or not dict(state.values):
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        state_dict = dict(state.values)
        return format_state_for_display(state_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# ENDPOINT 5: GET /sessions (bonus - for debugging)
# ================================================================

@router.get(
    "/sessions",
    summary="List all sessions",
    description="Returns all session IDs with conversation counts. For debugging.",
)
async def list_sessions():
    """Lists all sessions stored in long-term memory."""
    
    from memory.long_term import get_all_sessions
    sessions = get_all_sessions()
    return {"sessions": sessions, "count": len(sessions)}


# ================================================================
# ENDPOINT 6: POST /stream (streaming version)
# ================================================================

@router.post(
    "/stream",
    summary="Submit query with streaming trace updates",
    description=(
        "Same as /query but returns Server-Sent Events (SSE) so the "
        "frontend sees trace updates in real-time as each node runs. "
        "Each event is a JSON object with type and data fields."
    ),
)
async def stream_query(request: QueryRequest):
    """
    Streaming version of /query using Server-Sent Events.
    
    WHAT ARE SERVER-SENT EVENTS (SSE)?
    SSE is a protocol where the server sends multiple messages
    to the client over a single HTTP connection.
    
    Unlike WebSockets (bidirectional), SSE is one-way: server → client.
    Perfect for our use case: client submits query, server streams
    progress updates as the graph executes.
    
    FRONTEND USAGE:
        const eventSource = new EventSource('/api/v1/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateUI(data);  // update trace display in real-time
        };
    
    EACH EVENT CONTAINS:
        {
            "type": "trace",        // trace update
            "data": "Planner: ..."  // the trace message
        }
        {
            "type": "status",       // status change
            "data": "awaiting_approval"
        }
        {
            "type": "draft",        // draft is ready
            "data": "Based on recent research..."
        }
        {
            "type": "complete",     // all done
            "data": {"session_id": "..."}
        }
    """
    
    import json
    
    state = create_initial_state(
        query=request.query,
        session_id=request.session_id,
    )
    session_id = state["session_id"]
    
    # Load long-term context
    long_term_context = load_relevant_context(
        session_id=session_id,
        query=request.query,
    )
    if long_term_context:
        state["long_term_context"] = long_term_context
    
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()
    
    async def event_generator():
        """
        Async generator that yields SSE events.
        
        Each 'yield' sends one event to the frontend.
        The client sees it IMMEDIATELY, not after all events.
        
        This is the core of real-time streaming:
        instead of waiting for everything to finish,
        we send partial results as they become available.
        """
        
        # Send initial event with session_id
        yield {
            "event": "start",
            "data": json.dumps({
                "type": "start",
                "session_id": session_id,
                "query": request.query,
            })
        }
        
        # Run graph in thread and stream events
        loop = asyncio.get_event_loop()
        
        # We need a thread-safe way to stream events from the
        # synchronous graph to this async generator.
        # We use asyncio.Queue as a bridge:
        # - graph thread puts events into queue
        # - this generator reads from queue and yields them
        queue = asyncio.Queue()
        
        def run_graph_with_queue():
            """Runs in thread, puts events into queue."""
            for event in graph.stream(state, config):
                for node_name, node_state in event.items():
                    if node_name == "__interrupt__":
                        # Signal that we hit the checkpoint
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "checkpoint", "data": "awaiting_approval"}),
                            loop
                        )
                    elif "trace" in node_state:
                        for trace_entry in node_state.get("trace", []):
                            asyncio.run_coroutine_threadsafe(
                                queue.put({"type": "trace", "data": trace_entry}),
                                loop
                            )
                    
                    if "draft_answer" in node_state and node_state["draft_answer"]:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({
                                "type": "draft",
                                "data": node_state["draft_answer"]
                            }),
                            loop
                        )
            
            # Signal completion
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "done", "data": None}),
                loop
            )
        
        # Start graph in thread
        thread_future = loop.run_in_executor(None, run_graph_with_queue)
        
        # Yield events as they arrive
        while True:
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=120.0)
                
                if event_data["type"] == "done":
                    break
                
                yield {
                    "event": event_data["type"],
                    "data": json.dumps(event_data)
                }
                
            except asyncio.TimeoutError:
                # Send keepalive to prevent connection timeout
                yield {
                    "event": "keepalive",
                    "data": json.dumps({"type": "keepalive"})
                }
        
        # Wait for thread to finish
        await thread_future
        
        # Send final state
        current_state = graph.get_state(config)
        state_dict = dict(current_state.values)
        display = format_state_for_display(state_dict)
        
        yield {
            "event": "complete",
            "data": json.dumps({
                "type": "complete",
                "session_id": session_id,
                "status": display["status"],
                "tools_used": display.get("tools_used", []),
            })
        }
    
    return EventSourceResponse(event_generator())