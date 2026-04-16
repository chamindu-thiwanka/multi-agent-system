"""
nodes.py - LangGraph Node Functions
=====================================

PURPOSE:
    Each function here is one NODE in the LangGraph state machine.
    Nodes are the "steps" in our pipeline.

WHAT IS A NODE?
    A node is a Python function that:
    1. Receives the current GraphState
    2. Does some work (calls LLM, calls tool, etc.)
    3. Returns a PARTIAL state update (only the fields it changed)

    LangGraph automatically merges the partial update back into
    the full state before passing to the next node.

NODE FUNCTIONS IN THIS FILE:
    planner_node      - decides which tools to use
    tool_executor_node - calls the selected tools
    draft_node        - generates answer from tool results
    verifier_node     - checks answer quality
    human_checkpoint  - pauses for human review
    final_output_node - formats and saves final answer

THE PLANNER -> EXECUTOR -> VERIFIER PATTERN:
    This pattern mirrors how a thoughtful human researcher works:

    PLANNER:   "To answer this question, I should look at:
                recent research papers AND current news"
               (decides strategy before doing any work)

    EXECUTOR:  "OK, searching research papers... done.
                Searching internet... done.
                Here are all the results."
               (executes the strategy)

    VERIFIER:  "Does this answer the original question?
                Is it complete? Are there gaps?"
               (quality control before showing to human)

    This separation means each component has ONE clear job.
    If the verifier says the answer is incomplete, ONLY the
    planner and executor need to run again. The verifier itself
    doesn't need to change.
"""

import logging
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import GraphState
from config import get_llm, get_settings

logger = logging.getLogger(__name__)


# ================================================================
# HELPER: ADD TO TRACE
# ================================================================

def add_trace(message: str) -> list:
    """
    Helper to create a trace entry.

    Returns a list with one entry because our trace field uses
    Annotated[list, operator.add] - it ADDS lists together.
    So we always return a list, even for a single message.
    """
    logger.info(f"[Graph] {message}")
    return [message]


# ================================================================
# NODE 1: PLANNER
# ================================================================

def planner_node(state: GraphState) -> dict:
    """
    PLANNER NODE - Decides which tools to use for the query.

    WHAT IT DOES:
        Analyzes the user's question and creates a plan:
        - Which tools are needed? (RAG, search, Reddit)
        - Why? (reasoning)
        - Are we retrying? If so, what feedback to incorporate?

    WHY A DEDICATED PLANNER?
        Without a planner, you'd call ALL tools for EVERY query.
        That's slow and wasteful:
        - "What is a neural network?" needs only RAG
        - "Latest COVID news?" needs only internet search
        - "What do people think about Python vs R?" needs RAG + Reddit

        The planner makes smart decisions about resource usage.

    HOW IT DECIDES:
        We send the query to the LLM with a prompt that explains
        the available tools and asks it to choose.
        The LLM returns a structured JSON response:
        {
            "tools": ["rag", "internet_search"],
            "reasoning": "Question asks about research and recent news"
        }

    RETRY HANDLING:
        If retry_count > 0, we include the verifier's feedback
        so the planner can try a different approach.
    """

    query = state["query"]
    retry_count = state.get("retry_count", 0)
    verification_feedback = state.get("verification_feedback", "")
    long_term_context = state.get("long_term_context", "")

    trace_msg = (
        f"Planner: analyzing query (attempt {retry_count + 1})"
        if retry_count > 0
        else "Planner: analyzing query and selecting tools"
    )

    llm = get_llm()

    # Build the planning prompt
    # We ask for JSON output to make parsing reliable
    retry_instruction = ""
    if retry_count > 0 and verification_feedback:
        retry_instruction = (
            f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{verification_feedback}\n"
            f"Please adjust your approach based on this feedback."
        )

    context_instruction = ""
    if long_term_context:
        context_instruction = (
            f"\n\nUSER CONTEXT FROM PREVIOUS SESSIONS:\n{long_term_context}\n"
            f"Use this to personalize your approach."
        )

    planning_prompt = f"""You are a research planning agent. Given a user query,
decide which tools to use to answer it comprehensively.

AVAILABLE TOOLS:
1. "rag"             - Search PubMed research database (200 medical/AI papers)
                       Best for: academic questions, clinical research, 
                       scientific facts, in-depth technical topics
                       
2. "internet_search" - Search the live web (real-time results)
                       Best for: recent events, current news, general facts,
                       information newer than research database
                       
3. "reddit"          - Search Reddit community discussions
                       Best for: patient experiences, developer opinions,
                       community consensus, practical advice, debates

USER QUERY: {query}
{retry_instruction}
{context_instruction}

Respond with ONLY a JSON object in exactly this format, no other text:
{{
    "tools": ["rag", "internet_search"],
    "reasoning": "Brief explanation of why these tools were selected"
}}

Rules:
- tools must be a list containing one or more of: "rag", "internet_search", "reddit"
- Always include "rag" for medical or technical research questions
- Include "internet_search" for current events or recent developments
- Include "reddit" for opinion, experience, or community questions
- reasoning must be under 100 words"""

    try:
        response = llm.invoke([HumanMessage(content=planning_prompt)])
        response_text = response.content.strip()

        # Parse the JSON response from the LLM
        # We need to extract the JSON even if LLM adds extra text
        import json
        import re

        # Find JSON object in response
        # LLMs sometimes add text before/after the JSON
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)

        if json_match:
            plan_data = json.loads(json_match.group())
            tools = plan_data.get("tools", ["rag", "internet_search"])
            reasoning = plan_data.get("reasoning", "Using default tools")
        else:
            # If we can't parse JSON, use safe defaults
            logger.warning(
                f"Could not parse planner JSON response: {response_text[:200]}"
            )
            tools = ["rag", "internet_search"]
            reasoning = "Default plan (could not parse LLM response)"

        # Validate tools list - only allow known tool names
        valid_tools = {"rag", "internet_search", "reddit"}
        tools = [t for t in tools if t in valid_tools]
        if not tools:
            tools = ["rag", "internet_search"]

        logger.info(f"[Planner] Selected tools: {tools}")
        logger.info(f"[Planner] Reasoning: {reasoning}")

        return {
            "plan": tools,
            "plan_reasoning": reasoning,
            "retry_count": retry_count,
            "trace": add_trace(
                f"Planner: selected tools {tools} — {reasoning}"
            ),
        }

    except Exception as e:
        logger.error(f"Planner node failed: {e}")
        # Return safe defaults so graph doesn't crash
        return {
            "plan": ["rag", "internet_search"],
            "plan_reasoning": "Default plan (planner error)",
            "retry_count": retry_count,
            "trace": add_trace(f"Planner: error occurred, using defaults"),
            "error": str(e),
        }


# ================================================================
# NODE 2: TOOL EXECUTOR
# ================================================================

def tool_executor_node(state: GraphState) -> dict:
    """
    TOOL EXECUTOR NODE - Calls the tools selected by the planner.

    WHAT IT DOES:
        Reads the plan from state and calls each tool in sequence.
        Stores results back into state.

    WHY SEQUENTIAL (not parallel)?
        We call tools one at a time for simplicity and reliability.
        With local Ollama for embeddings, parallel calls could
        overwhelm the system. Sequential is safer and easier to debug.

        In production with cloud services, you'd run tools in parallel
        using asyncio.gather() for faster execution.

    ERROR HANDLING:
        If one tool fails, we log the error and continue with others.
        A single tool failure should not stop the whole pipeline.
        The draft node can still create an answer from partial results.
    """

    plan = state.get("plan", ["rag", "internet_search"])
    query = state["query"]

    logger.info(f"[Tool Executor] Executing plan: {plan}")

    # Import tools - done here to avoid circular imports
    # and to only import what's actually needed
    from tools.retrieval_tool import retrieve_documents
    from tools.search_tool import search_internet
    from tools.reddit_tool import search_reddit

    results = {}
    tools_used = []
    trace_entries = []

    # -- Execute RAG tool ---------------------------------------
    if "rag" in plan:
        trace_entries.append("Tool Executor: calling RAG retrieval tool")
        try:
            rag_result = retrieve_documents.invoke({
                "query": query,
                "n_results": 5,
            })
            results["rag_results"] = rag_result
            tools_used.append("rag")
            trace_entries.append(
                f"RAG Tool: retrieved research articles from PubMed database"
            )
            logger.info("[Tool Executor] RAG tool completed")
        except Exception as e:
            error_msg = f"RAG tool failed: {str(e)}"
            results["rag_results"] = error_msg
            trace_entries.append(f"RAG Tool: failed - {str(e)[:100]}")
            logger.error(f"[Tool Executor] RAG error: {e}")

    # -- Execute Internet Search tool --------------------------
    if "internet_search" in plan:
        trace_entries.append("Tool Executor: calling internet search tool")
        try:
            search_result = search_internet.invoke({
                "query": query,
                "max_results": 5,
            })
            results["search_results"] = search_result
            tools_used.append("internet_search")
            trace_entries.append(
                "Search Tool: retrieved current web results via Tavily"
            )
            logger.info("[Tool Executor] Search tool completed")
        except Exception as e:
            error_msg = f"Search tool failed: {str(e)}"
            results["search_results"] = error_msg
            trace_entries.append(f"Search Tool: failed - {str(e)[:100]}")
            logger.error(f"[Tool Executor] Search error: {e}")

    # -- Execute Reddit tool ------------------------------------
    if "reddit" in plan:
        trace_entries.append("Tool Executor: calling Reddit intelligence tool")
        try:
            reddit_result = search_reddit.invoke({
                "query": query,
                "max_posts": 5,
                "include_comments": True,
            })
            results["reddit_results"] = reddit_result
            tools_used.append("reddit")
            trace_entries.append(
                "Reddit Tool: retrieved community discussions and sentiment"
            )
            logger.info("[Tool Executor] Reddit tool completed")
        except Exception as e:
            error_msg = f"Reddit tool failed: {str(e)}"
            results["reddit_results"] = error_msg
            trace_entries.append(f"Reddit Tool: failed - {str(e)[:100]}")
            logger.error(f"[Tool Executor] Reddit error: {e}")

    return {
        **results,
        "tools_used": tools_used,
        "trace": trace_entries,
    }


# ================================================================
# NODE 3: DRAFT ANSWER GENERATOR
# ================================================================

def draft_node(state: GraphState) -> dict:
    """
    DRAFT NODE - Generates a comprehensive answer from tool results.

    WHAT IT DOES:
        Takes all tool results from state and asks the LLM to write
        a well-structured answer that:
        - Directly addresses the user's question
        - Synthesizes information from multiple sources
        - Includes proper citations (PMIDs, URLs)
        - Is appropriately detailed but not unnecessarily long

    WHY A DRAFT (not final)?
        The draft is shown to:
        1. The VERIFIER node (automated quality check)
        2. The HUMAN (final approval)

        This two-stage review ensures quality before the answer
        is presented as final. The word "draft" signals to the
        human that they should review it.

    CITATION FORMAT:
        We instruct the LLM to cite sources as:
        - Research papers: [Author Year, PMID: 12345678]
        - Web results:     [Source: website.com]
        - Reddit:          [Reddit: r/subreddit]
    """

    query = state["query"]
    rag_results = state.get("rag_results", "No research results available")
    search_results = state.get("search_results", "No web search results")
    reddit_results = state.get("reddit_results", "No Reddit results")
    long_term_context = state.get("long_term_context", "")
    verification_feedback = state.get("verification_feedback", "")

    llm = get_llm()

    # Include verifier feedback if retrying
    retry_instruction = ""
    if verification_feedback:
        retry_instruction = (
            f"\n\nIMPORTANT - PREVIOUS ATTEMPT WAS REJECTED:\n"
            f"{verification_feedback}\n"
            f"Please address these issues in your new answer."
        )

    context_section = ""
    if long_term_context:
        context_section = f"\n\nUSER BACKGROUND:\n{long_term_context}\n"

    draft_prompt = f"""You are a knowledgeable research assistant specializing in 
medical science and artificial intelligence.

Answer the following question using the provided sources.
Your answer should be comprehensive, accurate, and well-cited.

USER QUESTION: {query}
{context_section}
{retry_instruction}

RESEARCH DATABASE RESULTS (PubMed articles):
{rag_results}

CURRENT WEB SEARCH RESULTS:
{search_results}

REDDIT COMMUNITY DISCUSSIONS:
{reddit_results}

INSTRUCTIONS:
1. Write a clear, well-structured answer that directly addresses the question
2. Synthesize information from ALL provided sources
3. Cite sources inline using these formats:
   - Research papers: [PMID: 12345678] or [AuthorName Year]
   - Web sources: [Source: website-name]
   - Reddit discussions: [Reddit: r/subreddit-name]
4. If sources contradict each other, acknowledge this
5. Separate clearly what is established research vs recent news vs community opinion
6. Keep response focused and under 600 words
7. End with a "Sources" section listing key citations

Write your answer now:"""

    try:
        response = llm.invoke([HumanMessage(content=draft_prompt)])
        draft = response.content.strip()

        logger.info(
            f"[Draft Node] Draft generated ({len(draft)} characters)"
        )

        return {
            "draft_answer": draft,
            "trace": add_trace(
                f"Draft Generator: answer created using "
                f"{len([r for r in [rag_results, search_results, reddit_results] if r and 'failed' not in r.lower()])} "
                f"tool sources"
            ),
        }

    except Exception as e:
        error_msg = f"Draft generation failed: {str(e)}"
        logger.error(error_msg)
        return {
            "draft_answer": f"Error generating answer: {error_msg}",
            "trace": add_trace(f"Draft Generator: failed - {str(e)[:100]}"),
            "error": str(e),
        }


# ================================================================
# NODE 4: VERIFIER
# ================================================================

def verifier_node(state: GraphState) -> dict:
    """
    VERIFIER NODE - Evaluates the quality of the draft answer.

    WHAT IT DOES:
        Acts as a quality control check before human review.
        Asks the LLM to evaluate whether the draft:
        - Actually answers the question asked
        - Uses the provided sources (not just making things up)
        - Is complete (no obvious gaps)
        - Has citations

    WHY AUTOMATED VERIFICATION?
        Without this, every draft goes directly to human review.
        The verifier catches obvious failures automatically:
        - "The answer talks about something completely different"
        - "The answer just says 'I don't know'"
        - "The answer is 3 words long"

        This saves human review time and ensures basic quality.

    THE RETRY LOOP:
        If verification fails AND retry_count < 2:
            → graph goes BACK to planner
            → planner reads the feedback and tries differently
            → new draft is generated
            → verifier checks again

        If retry_count >= 2:
            → accept whatever draft we have
            → prevents infinite loops

    OUTPUT:
        Sets verification_passed = True/False
        Sets verification_feedback = detailed evaluation
        These control the graph's routing in edges.py
    """

    query = state["query"]
    draft_answer = state.get("draft_answer", "")
    retry_count = state.get("retry_count", 0)

    # Hard cap on retries to prevent infinite loops
    # If we've tried twice already, accept the current draft
    if retry_count >= 2:
        logger.info("[Verifier] Max retries reached, accepting draft")
        return {
            "verification_passed": True,
            "verification_feedback": "Accepted after maximum retries",
            "trace": add_trace(
                "Verifier: max retries reached, accepting current draft"
            ),
        }

    if not draft_answer or len(draft_answer) < 50:
        return {
            "verification_passed": False,
            "verification_feedback": (
                "Draft answer is too short or empty. "
                "Generate a more complete response."
            ),
            "retry_count": retry_count + 1,
            "trace": add_trace("Verifier: draft too short, requesting retry"),
        }

    llm = get_llm()

    verification_prompt = f"""You are a quality control reviewer for AI-generated answers.

Evaluate this draft answer and determine if it meets quality standards.

ORIGINAL QUESTION:
{query}

DRAFT ANSWER:
{draft_answer}

EVALUATION CRITERIA:
1. RELEVANCE: Does the answer directly address the question asked?
2. COMPLETENESS: Are there obvious important gaps or missing information?
3. CITATIONS: Does the answer cite sources (even approximately)?
4. COHERENCE: Is the answer well-structured and readable?
5. ACCURACY: Does anything seem factually suspicious or contradictory?

Respond with ONLY a JSON object in exactly this format:
{{
    "passed": true,
    "score": 8,
    "feedback": "Brief evaluation - what is good and what could improve"
}}

Rules:
- "passed" must be true or false
- "score" must be an integer 1-10
- "passed" should be true if score >= 6
- "feedback" must be under 150 words
- Be decisive - do not pass poor quality answers"""

    try:
        response = llm.invoke([HumanMessage(content=verification_prompt)])
        response_text = response.content.strip()

        import json
        import re

        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)

        if json_match:
            eval_data = json.loads(json_match.group())
            passed = eval_data.get("passed", True)
            score = eval_data.get("score", 7)
            feedback = eval_data.get("feedback", "Evaluation complete")
        else:
            # If can't parse, default to passing
            passed = True
            score = 7
            feedback = "Could not parse verification response, defaulting to pass"

        logger.info(
            f"[Verifier] Score: {score}/10 | "
            f"Passed: {passed} | "
            f"Feedback: {feedback[:100]}"
        )

        result = {
            "verification_passed": passed,
            "verification_feedback": feedback,
        }

        if not passed:
            result["retry_count"] = retry_count + 1
            result["trace"] = add_trace(
                f"Verifier: score {score}/10 - FAILED - retrying. "
                f"Feedback: {feedback[:100]}"
            )
        else:
            result["trace"] = add_trace(
                f"Verifier: score {score}/10 - PASSED. "
                f"Proceeding to human review."
            )

        return result

    except Exception as e:
        logger.error(f"Verifier node failed: {e}")
        # On error, pass the draft through
        return {
            "verification_passed": True,
            "verification_feedback": f"Verification error: {str(e)}",
            "trace": add_trace(
                f"Verifier: error occurred, passing draft through"
            ),
        }


# ================================================================
# NODE 5: HUMAN CHECKPOINT
# ================================================================

def human_checkpoint_node(state: GraphState) -> dict:
    """
    HUMAN CHECKPOINT NODE - Prepares state for human review pause.

    IMPORTANT: This node does NOT actually pause the graph.
    The PAUSE is configured in graph.py using interrupt_before.
    LangGraph pauses BEFORE executing this node, saves state,
    and waits for the /approve API call to resume.

    So this node only runs AFTER the human has approved.

    WHAT HAPPENS:
    1. Graph reaches this node
    2. LangGraph sees interrupt_before=["human_checkpoint"]
    3. LangGraph saves complete state to SQLite checkpointer
    4. Graph execution stops
    5. FastAPI returns draft to frontend
    6. Human reads draft, clicks Approve
    7. Frontend calls POST /approve
    8. FastAPI resumes graph from checkpoint
    9. THIS NODE FINALLY EXECUTES
    10. Sets human_approved = True
    11. Graph continues to final_output_node

    WHY IMPLEMENT IN GRAPH STATE (not UI logic)?
        The assignment specifically requires this.
        If it were UI logic only, the backend wouldn't know
        the human approved - it would just show a button.
        Implementing in graph state means the backend enforces
        the approval requirement at the architectural level.
    """

    logger.info("[Human Checkpoint] Human has approved the draft")

    return {
        "human_approved": True,
        "trace": add_trace(
            "Human Checkpoint: draft approved by human reviewer"
        ),
    }


# ================================================================
# NODE 6: FINAL OUTPUT
# ================================================================

def final_output_node(state: GraphState) -> dict:
    """
    FINAL OUTPUT NODE - Produces the polished final answer.

    WHAT IT DOES:
        Takes the human-approved draft and:
        1. Formats it cleanly with proper structure
        2. Saves a summary to long-term memory (SQLite)
        3. Returns the final answer

    WHY A SEPARATE FINAL NODE?
        The draft is designed for review - it may be verbose,
        have rough formatting, or include evaluation notes.
        The final output is designed for presentation to the user.

        Also, this is where we:
        - Record what was asked and answered (long-term memory)
        - Clean up any formatting issues
        - Add metadata (session info, sources used)

    LONG-TERM MEMORY:
        After each conversation, we save a summary:
        "User asked about Alzheimer's AI diagnosis.
         Answer covered DL approaches, recent FDA approvals,
         and patient community perspectives."

        Next time the user asks something related, we load this
        context so the agent can provide personalized, continuous
        assistance.
    """

    query = state["query"]
    draft_answer = state.get("draft_answer", "No answer generated")
    tools_used = state.get("tools_used", [])
    session_id = state["session_id"]

    # Format the final answer
    # We enhance the draft with metadata about sources used
    tool_descriptions = {
        "rag": "PubMed Research Database",
        "internet_search": "Live Web Search",
        "reddit": "Reddit Community Discussions",
    }

    sources_note = ""
    if tools_used:
        source_list = [tool_descriptions.get(t, t) for t in tools_used]
        sources_note = (
            f"\n\n---\n"
            f"*Sources used: {', '.join(source_list)}*"
        )

    final_answer = draft_answer + sources_note

    # Save to long-term memory
    try:
        from memory.long_term import save_conversation_summary
        save_conversation_summary(
            session_id=session_id,
            query=query,
            answer_summary=draft_answer[:500],  # save first 500 chars as summary
            tools_used=tools_used,
        )
        logger.info(f"[Final Output] Saved conversation to long-term memory")
    except Exception as e:
        logger.warning(f"[Final Output] Could not save to memory: {e}")

    logger.info(f"[Final Output] Final answer ready ({len(final_answer)} chars)")

    return {
        "final_answer": final_answer,
        "trace": add_trace(
            f"Final Output: answer complete using {tools_used}. "
            f"Conversation saved to memory."
        ),
    }