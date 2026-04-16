"""
long_term.py - Long-Term Memory Storage
=========================================

PURPOSE:
    Persists conversation summaries to SQLite so the agent
    remembers past interactions across different sessions.

SHORT-TERM vs LONG-TERM MEMORY:

    SHORT-TERM (GraphState.messages):
        - Lives in graph state during ONE session
        - Gone when the session ends
        - Like your working memory while solving a problem

    LONG-TERM (this module, SQLite):
        - Persists across sessions indefinitely
        - Loaded at the start of each new session
        - Like your memory of past conversations

WHY SQLITE?
    - Built into Python (no installation needed)
    - Single file (easy to backup, no server to manage)
    - Fast enough for our use case
    - Can store thousands of conversations

THE DATABASE SCHEMA:
    Table: conversations
    Columns:
        id          - auto-increment primary key
        session_id  - which session this came from
        query       - what the user asked
        answer_summary - first 500 chars of the answer
        tools_used  - which tools were called (JSON list)
        created_at  - when this was saved

PRIVACY NOTE:
    In a real system, you'd need user consent to store conversations.
    For this assignment, we're building the technical capability.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Creates a connection to the SQLite database.

    check_same_thread=False allows the connection to be used
    from different threads (FastAPI uses async threads).

    The database file is created automatically if it doesn't exist.
    SQLite is serverless - the "server" IS the file.
    """
    settings = get_settings()
    db_path = settings.long_term_memory_path

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # row_factory=sqlite3.Row means query results are dict-like
    # instead of plain tuples. So you can do row["query"] instead
    # of row[1]. Much more readable.
    return conn


def initialize_memory_db():
    """
    Creates the conversations table if it doesn't exist.

    We use IF NOT EXISTS so this is safe to call multiple times.
    Call this once at application startup.

    SQL CREATE TABLE breakdown:
        id INTEGER PRIMARY KEY AUTOINCREMENT
            - Unique auto-incrementing number for each row
        session_id TEXT NOT NULL
            - Which session this belongs to (our UUID)
        query TEXT NOT NULL
            - The user's original question
        answer_summary TEXT
            - First 500 chars of the answer (not full answer)
        tools_used TEXT DEFAULT '[]'
            - JSON-encoded list: '["rag", "internet_search"]'
        created_at TEXT
            - ISO format datetime string: "2024-01-15T10:30:00"
    """

    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                answer_summary TEXT,
                tools_used TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)

        # Index on session_id for fast lookups
        # Without an index, loading history would scan ALL rows
        # With an index, it jumps directly to matching rows
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON conversations(session_id)
        """)

        conn.commit()
        logger.info("Long-term memory database initialized")
    finally:
        conn.close()


def save_conversation_summary(
    session_id: str,
    query: str,
    answer_summary: str,
    tools_used: list,
):
    """
    Saves a conversation summary to long-term memory.

    Called by final_output_node after each successful conversation.

    WHY SUMMARY NOT FULL ANSWER?
        Full answers can be thousands of words. Loading 20 full
        answers as context would exceed the LLM's context window.
        A 500-character summary gives the agent enough to know
        what was discussed without overwhelming its context.
    """

    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO conversations 
            (session_id, query, answer_summary, tools_used, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            query,
            answer_summary[:500],  # cap at 500 chars
            json.dumps(tools_used),
            datetime.now().isoformat(),
        ))
        conn.commit()
        logger.info(
            f"Saved conversation to long-term memory "
            f"(session: {session_id[:8]}...)"
        )
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
    finally:
        conn.close()


def load_relevant_context(session_id: str, query: str) -> str:
    """
    Loads recent conversation history as context for a new query.

    WHY LOAD CONTEXT?
        If the user previously asked about Parkinson's disease,
        and now asks "what are the latest treatment updates?",
        the agent should know to focus on Parkinson's treatments
        without the user having to repeat themselves.

        This is what makes the system feel like a continuous
        assistant rather than starting fresh each time.

    WHAT WE LOAD:
        The last N conversations (from settings.memory_lookback_count)
        for this session_id. We format them as a summary string
        that gets injected into the planner and draft prompts.

    FUTURE IMPROVEMENT:
        Use semantic similarity to load RELEVANT past conversations,
        not just the most recent ones. This would require embedding
        past queries and finding the most similar to the current one.
        For this assignment, recency-based loading is sufficient.
    """

    settings = get_settings()
    lookback = settings.memory_lookback_count

    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT query, answer_summary, tools_used, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, lookback)).fetchall()

        if not rows:
            return ""

        # Format as readable context string
        context_lines = [
            f"Previous conversations in this session ({len(rows)} found):"
        ]

        # Reverse so oldest is first (chronological order reads better)
        for row in reversed(rows):
            tools = json.loads(row["tools_used"])
            context_lines.append(
                f"- Asked: \"{row['query']}\"\n"
                f"  Summary: {row['answer_summary'][:200]}\n"
                f"  Tools used: {', '.join(tools)}"
            )

        context = "\n".join(context_lines)
        logger.info(
            f"Loaded {len(rows)} past conversations for session "
            f"{session_id[:8]}..."
        )
        return context

    except Exception as e:
        logger.error(f"Failed to load context: {e}")
        return ""
    finally:
        conn.close()


def get_all_sessions() -> list:
    """
    Returns a list of all unique session IDs.
    Used for debugging and admin purposes.
    """
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT DISTINCT session_id, 
                   COUNT(*) as conversation_count,
                   MAX(created_at) as last_active
            FROM conversations
            GROUP BY session_id
            ORDER BY last_active DESC
        """).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()