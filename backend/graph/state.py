"""
state.py - LangGraph State Definition
======================================

PURPOSE:
    Defines GraphState - the shared data structure that flows through
    every node in our LangGraph pipeline.

WHAT IS GRAPH STATE?
    Imagine a relay race. Each runner (node) receives a baton (state),
    does their job, and passes the updated baton to the next runner.
    The baton contains everything accumulated so far.

    In LangGraph, state is a TypedDict (typed dictionary).
    Every node receives the FULL state and returns ONLY the fields
    it changed. LangGraph merges the changes back automatically.

    Example:
        Planner node receives:  {query: "...", plan: None, ...}
        Planner node returns:   {plan: ["rag", "search"], trace: [...]}
        LangGraph merges:       {query: "...", plan: ["rag", "search"], ...}

WHY TypedDict INSTEAD OF A REGULAR DICT?
    TypedDict gives us type hints - Python knows what fields exist
    and what types they should be. This means:
    - Better IDE autocomplete
    - Clearer documentation
    - Catch bugs early (if you write state["typo"], IDE warns you)

WHY Optional[] FOR MOST FIELDS?
    When a graph run starts, most fields are empty.
    Optional[str] means the field can be either str or None.
    This lets us initialize state with just the query and session_id,
    and fill in other fields as the graph progresses.

ANNOTATION WITH Annotated[list, operator.add]:
    Used for the "messages" and "trace" fields.
    This tells LangGraph: "when merging state, ADD to this list
    instead of replacing it."

    Without annotation:
        Node A sets trace = ["step 1"]
        Node B sets trace = ["step 2"]
        Result: trace = ["step 2"]  ← step 1 lost!

    With Annotated[list, operator.add]:
        Node A sets trace = ["step 1"]
        Node B sets trace = ["step 2"]
        Result: trace = ["step 1", "step 2"]  ← both preserved!
"""

import operator
from typing import Optional, Annotated
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    The complete state of one agent pipeline run.

    This dictionary travels through every node in the graph.
    Fields are filled in progressively as each node executes.

    FIELD GROUPS:
    1. Input fields      - set at the start, never change
    2. Planner fields    - set by the planner node
    3. Tool result fields - set by the tool executor node
    4. Draft fields      - set by the draft generator node
    5. Verification fields - set by the verifier node
    6. Approval fields   - set by human checkpoint
    7. Output fields     - set by final output node
    8. Memory fields     - used throughout for context
    9. Observability     - trace log visible in UI
    """

    # ----------------------------------------------------------
    # 1. INPUT FIELDS
    # Set once at the start of each run, never modified.
    # ----------------------------------------------------------

    query: str
    # The user's original question.
    # Example: "What are latest treatments for Parkinson's disease?"

    session_id: str
    # Unique ID for this conversation session.
    # Used to:
    # - Look up long-term memory (past conversations)
    # - Save checkpoint state for human-in-the-loop resume
    # - Track which run corresponds to which /approve call

    # ----------------------------------------------------------
    # 2. PLANNER FIELDS
    # Set by the planner node after analyzing the query.
    # ----------------------------------------------------------

    plan: Optional[list]
    # Which tools to use for this query.
    # Example: ["rag", "internet_search"]
    # Example: ["rag", "internet_search", "reddit"]
    # Example: ["rag"]  (simple factual question, RAG sufficient)

    plan_reasoning: Optional[str]
    # Why the planner chose these tools.
    # Example: "Question asks about recent developments, so internet
    #           search needed alongside research database"
    # Shown in UI trace for transparency.

    # ----------------------------------------------------------
    # 3. TOOL RESULT FIELDS
    # Set by the tool executor node after calling each tool.
    # Each field stores the formatted string returned by its tool.
    # ----------------------------------------------------------

    rag_results: Optional[str]
    # Results from retrieve_documents() tool.
    # Contains article titles, authors, excerpts, PMIDs, URLs.

    search_results: Optional[str]
    # Results from search_internet() tool.
    # Contains titles, URLs, content snippets from web search.

    reddit_results: Optional[str]
    # Results from search_reddit() tool.
    # Contains post summaries, sentiment analysis, community themes.

    tools_used: Optional[list]
    # Which tools were actually called (may differ from plan if tool fails).
    # Example: ["rag", "internet_search"]  (reddit failed/skipped)
    # Used in UI trace and final answer attribution.

    # ----------------------------------------------------------
    # 4. DRAFT FIELDS
    # Set by the draft answer generator node.
    # ----------------------------------------------------------

    draft_answer: Optional[str]
    # The LLM's first attempt at answering the query.
    # Shown to human for review before final output.
    # May be revised if verifier rejects it.

    # ----------------------------------------------------------
    # 5. VERIFICATION FIELDS
    # Set by the verifier node after evaluating the draft.
    # ----------------------------------------------------------

    verification_passed: Optional[bool]
    # True if the verifier approved the draft answer.
    # False if the answer needs improvement.
    # Controls whether graph continues or loops back to planner.

    verification_feedback: Optional[str]
    # The verifier's evaluation of the draft.
    # If passed: brief confirmation of quality
    # If failed: specific feedback on what needs improvement
    # This feedback is passed back to the planner if retrying.

    retry_count: Optional[int]
    # How many times the planner-executor-verifier loop has run.
    # We cap this at 2 to prevent infinite loops.
    # If retry_count >= 2, we accept whatever draft we have.

    # ----------------------------------------------------------
    # 6. APPROVAL FIELDS
    # Set by the human checkpoint and /approve endpoint.
    # ----------------------------------------------------------

    human_approved: Optional[bool]
    # False: graph is paused, waiting for human decision
    # True:  human clicked Approve, graph should continue
    # None:  not yet reached the checkpoint

    human_feedback: Optional[str]
    # Optional feedback the human provides when approving/rejecting.
    # If they reject, this explains what to change.
    # Currently we only support approve (not reject + revise).

    # ----------------------------------------------------------
    # 7. OUTPUT FIELDS
    # Set by the final output node.
    # ----------------------------------------------------------

    final_answer: Optional[str]
    # The polished, final answer shown to the user.
    # Formatted with proper citations and structure.
    # Only set after human approval.

    # ----------------------------------------------------------
    # 8. MEMORY FIELDS
    # Used for maintaining context within and across sessions.
    # ----------------------------------------------------------

    long_term_context: Optional[str]
    # Summary of relevant past conversations loaded at session start.
    # Gives the agent context about what this user has asked before.
    # Example: "In previous sessions, user asked about Parkinson's
    #           treatment options and ML diagnostic tools."

    messages: Annotated[list, operator.add]
    # Conversation history for this session.
    # Uses Annotated[list, operator.add] so messages ACCUMULATE
    # across node calls instead of being overwritten.
    # Each message: {"role": "user"/"assistant", "content": "..."}

    # ----------------------------------------------------------
    # 9. OBSERVABILITY FIELDS
    # Trace log shown in the UI so users can see what happened.
    # ----------------------------------------------------------

    trace: Annotated[list, operator.add]
    # Step-by-step log of what the graph did.
    # Uses Annotated[list, operator.add] so entries ACCUMULATE.
    # Each entry is a string describing one step.
    # Example entries:
    #   "Planner: decided to use RAG + internet search"
    #   "RAG Tool: retrieved 5 relevant articles from PubMed"
    #   "Search Tool: found 3 results from Tavily"
    #   "Verifier: quality check passed"
    #   "Checkpoint: awaiting human approval"
    # Shown in real-time in the frontend UI.

    error: Optional[str]
    # If something goes wrong, error message is stored here.
    # Allows graceful error handling instead of crashes.
    # Frontend shows this to user if set.