# Design Decisions

## 1. Orchestration Decisions

### Why LangGraph Over Simple Sequential Code

A linear Python script cannot implement the requirements of this system.
LangGraph's state machine enables three things a script cannot do:

**Looping:** The verifier node can route back to the planner if answer
quality is insufficient (score < 6/10). This retry loop requires a graph —
a script would simply execute once and return.

**Pausing:** The human-in-the-loop checkpoint stops graph execution
completely, serializes all state to SQLite, and waits for an external
`/approve` API call to resume. This is implemented via LangGraph's
`interrupt_before` parameter — not simulated in the UI layer.

**Branching:** The planner dynamically selects which tools to run based
on query analysis. Some queries need only RAG; others need all three tools.
Conditional edges implement this routing cleanly.

### The Planner → Executor → Verifier Pattern

This three-stage pattern is a deliberate architectural choice:

- **Planner** separates decision-making from execution. It analyzes the
  query and selects tools before any expensive operations run.
- **Executor** calls tools and accumulates results. Isolated from both
  planning logic and answer generation.
- **Verifier** performs automated quality control before human review,
  reducing the burden on human reviewers by catching obvious failures.

### Structured Outputs From the LLM

Both the Planner and Verifier nodes instruct the LLM to respond with
JSON only. A `re.search(r'\{[^{}]*\}', response)` extracts valid JSON
even if the LLM adds preamble text. This makes agent outputs reliable
and machine-parseable regardless of which model is configured.

---

## 2. Memory Design

### Two-Tier Memory Architecture

**Short-term memory (GraphState):**
Lives in memory during a single graph execution. Every node reads from
and writes to GraphState — it is the shared "whiteboard" for one session.
Implemented as a LangGraph TypedDict with `Annotated[list, operator.add]`
for accumulating fields (trace, messages) that should append rather than overwrite.

**Long-term memory (SQLite):**
Persists conversation summaries across sessions in `memory_store.db`.
At the start of each new session, the last N conversations are loaded
as context for the planner and draft nodes. This gives the agent
continuity — it knows what the user has asked before.

**Checkpoint storage (SQLite):**
`graph_checkpoints.db` stores complete graph state snapshots for the
human-in-the-loop mechanism. When the graph pauses, LangGraph
serializes the entire state (all tool results, the draft, trace) to this
database keyed by `thread_id` (session_id). Resuming loads this snapshot
and continues from exactly where execution stopped.

### Why SQLite for Both Memory Stores

SQLite requires zero infrastructure — it is a file, not a server.
For a system already requiring Docker (Chroma), Ollama, and the
backend, adding a PostgreSQL or Redis service would increase setup
complexity significantly. SQLite is a deliberate scope trade-off.
In production, both stores would migrate to a proper database.

---

## 3. Extensibility Considerations

### Model-Agnostic Design

All LLM and embedding model creation is centralized in `config.py`
behind `get_llm()` and `get_embeddings()` factory functions.
No file in the codebase imports `ChatOllama` or `ChatOpenAI` directly.
Changing the model requires editing one line in `.env`:

```bash
LLM_MODEL_NAME=mistral:latest  # change this to switch models
```

Adding a new provider (e.g., Groq, Cohere) requires adding one
`elif` branch to `config.py`. All agents, nodes, and tools
automatically use the new provider.

### Tool Adapter Pattern

Each tool is a standalone LangChain `@tool` function in its own file.
Adding a new tool (e.g., ArXiv search, Wikipedia) requires:
1. Create `backend/tools/arxiv_tool.py` with `@tool` decorator
2. Import and call it in `tool_executor_node()` in `nodes.py`
3. Add it to the planner's prompt as an available option

No changes to the graph structure, routing logic, or API are needed.

### Swappable Vector Database

The RAG tool connects to Chroma via `chromadb.HttpClient()`.
Switching to a different vector database (Qdrant, Weaviate, Pinecone)
requires only modifying `tools/retrieval_tool.py` — the tool's
interface to the rest of the system remains identical.

### Adding New Agent Types

The current three agents (RAG, search, Reddit) are all implemented as
tool adapters called by a single `tool_executor_node`. Adding a new agent
type (e.g., a SQL database agent, a code execution agent) requires adding
a new tool file and updating the planner prompt. The LangGraph structure
does not change.

---

## 4. Simplifying Assumptions (Documented)

Per assignment guidance, assumptions are documented here:

- **Reddit API pending approval:** Reddit ended self-service API access.
  Credentials are configured; the tool will activate automatically
  when API access is approved. The system degrades gracefully.

- **Abstracts only from PubMed:** Full-text access requires publisher
  agreements. Abstracts provide sufficient content for demonstrating
  hybrid RAG retrieval.

- **Single-user assumption:** The SQLite checkpointer uses `session_id`
  as `thread_id`. Multi-user concurrent sessions work correctly because
  each session gets a unique UUID. However, no authentication layer
  is implemented (explicitly out of scope per assignment).

- **Local LLM response time:** With Ollama running locally, queries take
  2-10 minutes. In production with cloud LLMs this reduces to 10-30 seconds.

- **Tavily over Google Search API:** Tavily provides equivalent functionality
  with a simpler integration, free tier, and native LangChain support.