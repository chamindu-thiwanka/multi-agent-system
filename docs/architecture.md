# System Architecture

## Overview

Multi-Agent Research System built with LangGraph, FastAPI, Chroma, and Ollama.
Answers complex research queries by combining PubMed RAG, internet search,
and Reddit intelligence with human-in-the-loop approval.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
│                   http://localhost:3000                          │
│                                                                  │
│  [Query Input]  →  [Progress Bar]  →  [Trace + Draft]           │
│                                         ↓                        │
│                                    [Approve Button]              │
│                                         ↓                        │
│                                    [Final Answer]                │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP POST /api/v1/query
                       │ HTTP POST /api/v1/approve
                       │ HTTP GET  /api/v1/trace
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BACKEND SERVICE  :8000                          │
│                  FastAPI + Uvicorn                               │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              LANGGRAPH STATE MACHINE                       │  │
│  │                                                            │  │
│  │   ┌──────────┐    ┌──────────────┐    ┌───────────┐      │  │
│  │   │ PLANNER  │───▶│ TOOL         │───▶│  DRAFT    │      │  │
│  │   │          │    │ EXECUTOR     │    │ GENERATOR │      │  │
│  │   │ Decides  │    │              │    │           │      │  │
│  │   │ which    │    │ ┌──────────┐ │    │ LLM writes│      │  │
│  │   │ tools    │    │ │RAG Tool  │ │    │ answer    │      │  │
│  │   │ to use   │    │ │Search    │ │    │ from all  │      │  │
│  │   │          │    │ │Tool      │ │    │ sources   │      │  │
│  │   │ LLM      │    │ │Reddit    │ │    └─────┬─────┘      │  │
│  │   │ reasoning│    │ │Tool      │ │          │            │  │
│  │   └──────────┘    │ └──────────┘ │          ▼            │  │
│  │        ▲          └──────────────┘    ┌───────────┐      │  │
│  │        │ [retry if                    │ VERIFIER  │      │  │
│  │        │  score < 6]                  │           │      │  │
│  │        └──────────────────────────────│ LLM checks│      │  │
│  │                                       │ quality   │      │  │
│  │                                       │ score/10  │      │  │
│  │                                       └─────┬─────┘      │  │
│  │                                             │             │  │
│  │                                    [score >= 6]           │  │
│  │                                             ▼             │  │
│  │                                  ┌─────────────────────┐  │  │
│  │                                  │  HUMAN CHECKPOINT   │  │  │
│  │                                  │                     │  │  │
│  │                                  │  ◀── GRAPH PAUSES   │  │  │
│  │                                  │  State saved to     │  │  │
│  │                                  │  SQLite             │  │  │
│  │                                  │  Waits for          │  │  │
│  │                                  │  POST /approve      │  │  │
│  │                                  └──────────┬──────────┘  │  │
│  │                                             │             │  │
│  │                                    [human approved]       │  │
│  │                                             ▼             │  │
│  │                                  ┌─────────────────────┐  │  │
│  │                                  │  FINAL OUTPUT       │  │  │
│  │                                  │  Format + Citations  │  │  │
│  │                                  │  Save to memory     │  │  │
│  │                                  └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    MEMORY LAYER                          │    │
│  │                                                          │    │
│  │  Short-term: GraphState (in-memory during session)       │    │
│  │  Long-term:  SQLite (memory_store.db, persists forever)  │    │
│  │  Checkpoint: SQLite (graph_checkpoints.db, for resume)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────┬────────────────────┬────────────────────────-┘
                   │                    │
         ┌─────────▼──────┐   ┌────────▼────────┐
         │  CHROMA :8001  │   │  OLLAMA :11434  │
         │  (Docker)      │   │  (Local)        │
         │                │   │                 │
         │  434+ document │   │  llama3:latest  │
         │  chunks from   │   │  mistral:latest │
         │  PubMed        │   │  qwen3:1.7b     │
         │                │   │  nomic-embed-   │
         │  Vector DB for │   │  text           │
         │  RAG retrieval │   │                 │
         └────────────────┘   └─────────────────┘
                   │
         ┌─────────▼─────────────────────────────┐
         │           EXTERNAL APIs               │
         │                                       │
         │  Tavily Search API (internet search)  │
         │  PubMed / NCBI E-utilities (ingestion)│
         │  Reddit API (community intelligence)  │
         └───────────────────────────────────────┘
```

---

## Service Map (Docker Compose)

```mermaid
graph TB
    Browser["🌐 Browser<br/>localhost:3000"] --> Frontend
    Frontend["📄 Frontend<br/>nginx:alpine<br/>port 3000"] --> Backend
    Backend["⚙️ Backend<br/>FastAPI + LangGraph<br/>port 8000"] --> Chroma
    Backend --> Ollama
    Backend --> Tavily
    Backend --> Reddit
    Backend --> PubMed
    Chroma["🗄️ Chroma<br/>Vector DB<br/>port 8001<br/>(Docker)"]
    Ollama["🤖 Ollama<br/>Local LLM<br/>port 11434<br/>(Host)"]
    Tavily["🔍 Tavily<br/>Search API<br/>(Cloud)"]
    Reddit["💬 Reddit API<br/>Community<br/>(Cloud)"]
    PubMed["📚 PubMed<br/>NCBI API<br/>(Cloud)"]
```

---

## LangGraph State Machine

```mermaid
stateDiagram-v2
    [*] --> planner_node : User submits query

    planner_node --> tool_executor_node : Plan created

    tool_executor_node --> draft_node : Tools executed\n(RAG + Search + Reddit)

    draft_node --> verifier_node : Draft generated

    verifier_node --> planner_node : Score below 6\n(retry, max 2x)
    verifier_node --> human_checkpoint_node : Score >= 6\n✓ PASSED

    human_checkpoint_node --> human_checkpoint_node : PAUSED\nAwaiting /approve call

    human_checkpoint_node --> final_output_node : Human approved ✓

    final_output_node --> [*] : Final answer\n+ memory saved
```

---

## Data Flow: RAG Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Planner
    participant RAGTool
    participant Chroma
    participant Ollama
    participant Draft

    User->>Planner: "What are AI approaches for Alzheimer's?"
    Planner->>Planner: LLM decides to use RAG + internet_search
    Planner->>RAGTool: invoke(query)
    RAGTool->>Ollama: embed_query(query) → vector[768]
    Ollama-->>RAGTool: [0.23, -0.45, 0.87, ...]
    RAGTool->>Chroma: query(vector, where={filters}, n=5)
    Chroma-->>RAGTool: Top 5 matching chunks + metadata
    RAGTool-->>Planner: Formatted article excerpts with PMIDs
    Planner->>Draft: All tool results
    Draft->>Ollama: Generate answer using sources
    Ollama-->>Draft: Comprehensive answer with citations
```

---

## Folder Structure

```
multi-agent-system/
│
├── docker-compose.yml          # Orchestrates all services
├── .env                        # Real secrets (not in Git)
├── .env.example                # Template (in Git)
├── README.md                   # Setup and usage guide
│
├── docs/
│   ├── architecture.md         # This file
│   └── design.md               # Design decisions
│
├── backend/                    # FastAPI + LangGraph
│   ├── main.py                 # FastAPI app + startup
│   ├── config.py               # Model-agnostic config
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Backend container
│   │
│   ├── graph/                  # LangGraph orchestration
│   │   ├── state.py            # GraphState TypedDict
│   │   ├── nodes.py            # 6 node functions
│   │   ├── edges.py            # Routing/conditional logic
│   │   └── graph.py            # Graph assembly + compile
│   │
│   ├── tools/                  # Tool adapters
│   │   ├── retrieval_tool.py   # Chroma hybrid RAG tool
│   │   ├── search_tool.py      # Tavily internet search
│   │   └── reddit_tool.py      # Reddit PRAW tool
│   │
│   ├── agents/                 # Agent configurations
│   ├── memory/                 # Memory management
│   │   ├── short_term.py       # GraphState utilities
│   │   └── long_term.py        # SQLite persistence
│   │
│   └── api/
│       └── routes.py           # /query /approve /trace
│
├── frontend/                   # HTML + JavaScript UI
│   ├── index.html              # Complete single-file UI
│   └── Dockerfile              # nginx container
│
├── scripts/
│   └── ingest_pubmed.py        # Knowledge base ingestion
│
└── chroma_data/                # Chroma vector DB data
``` 