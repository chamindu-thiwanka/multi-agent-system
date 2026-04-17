# Multi-Agent Research System

A production-structured multi-agent AI system built with LangGraph,
FastAPI, Chroma, and Ollama. Answers complex research queries by
combining PubMed RAG, live internet search, and Reddit community
intelligence with human-in-the-loop approval.

---

## What This System Does

You ask a research question. The system:

1. **Plans** which tools to use (LLM decides based on query type)
2. **Searches** your PubMed knowledge base (434+ medical/AI articles)
3. **Searches** the live web via Tavily
4. **Checks Reddit** for community perspectives
5. **Drafts** a comprehensive answer with citations
6. **Verifies** quality automatically (scores 1-10, retries if < 6)
7. **Pauses** for your review before finalizing
8. **Delivers** a polished final answer after your approval

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for full diagrams.

**Key components:**

```
Browser (port 3000)
    ↓ HTTP
FastAPI Backend (port 8000)
    ↓ LangGraph State Machine
    ├── Planner Node       (LLM decides tools)
    ├── Tool Executor      (RAG + Search + Reddit)
    ├── Draft Generator    (LLM writes answer)
    ├── Verifier           (LLM quality check)
    ├── Human Checkpoint   ← PAUSES HERE
    └── Final Output       (formats + saves)
    ↓
Chroma Vector DB (port 8001, Docker)
Ollama Local LLM (port 11434)
```

---

## Prerequisites

- Python 3.10+
- Docker Desktop (for Chroma)
- [Ollama](https://ollama.ai) with these models:

```bash
ollama pull llama3:latest
ollama pull mistral:latest
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
```

- Free API keys:
  - [Tavily](https://tavily.com) — internet search
  - [NCBI](https://www.ncbi.nlm.nih.gov/account/) — PubMed access
  - [Reddit](https://www.reddit.com/prefs/apps) — community intelligence

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chamindu-thiwanka/multi-agent-system.git
cd multi-agent-system
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # Mac/Linux
```

Edit `.env` and fill in your API keys:

```bash
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama3:latest   # or mistral:latest for faster responses
TAVILY_API_KEY=your_key_here
NCBI_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
```

### 5. Start Chroma (vector database)

```bash
docker-compose up chroma -d
```

### 6. Ingest the knowledge base (run once)

```bash
python scripts/ingest_pubmed.py
```

This fetches 200 research articles from PubMed (100 per topic)
and stores them as searchable embeddings in Chroma.
Takes approximately 5-10 minutes.

### 7. Start the backend

```bash
cd backend
python main.py
```

Backend available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 8. Start the frontend (new terminal)

```bash
cd frontend
python -m http.server 3000
```

Frontend available at: `http://localhost:3000`

---

## Running with Docker (Full Stack)

```bash
docker-compose up
```

All services start automatically. Visit `http://localhost:3000`.

---

## Example Queries

These queries demonstrate different tool combinations:

**RAG + Internet Search:**
```
What are the latest developments in AI for Parkinson's disease?
What machine learning approaches detect Alzheimer's early?
How accurate are deep learning models for brain MRI analysis?
```

**RAG + Internet Search + Reddit:**
```
How is deep learning used in neurological disorder diagnosis?
What do researchers think about AI-assisted diagnosis tools?
What are the practical challenges of ML in clinical settings?
```

**RAG-focused (deep technical):**
```
What biomarkers are used in neurological disorder classification?
Compare CNN and transformer approaches for medical imaging?
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Submit a research query. Returns draft + trace after processing. |
| POST | `/api/v1/approve` | Approve the draft. Returns final answer. Requires `session_id`. |
| GET | `/api/v1/trace?session_id=...` | Get execution trace for a session. |
| GET | `/api/v1/status/{session_id}` | Get full session status. |
| GET | `/api/v1/sessions` | List all sessions. |
| GET | `/health` | Health check. |

### Example: Full Query Flow

```bash
# Step 1: Submit query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are AI approaches for Alzheimer detection?"}'

# Returns: {"session_id": "abc123", "draft_answer": "...", "trace": [...]}

# Step 2: Approve
curl -X POST http://localhost:8000/api/v1/approve \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "approved": true}'

# Returns: {"final_answer": "...", "status": "complete"}
```

---

## Switching Models

Change `LLM_MODEL_NAME` in `.env` — no code changes required:

```bash
# Fastest (for development/testing)
LLM_MODEL_NAME=qwen3:1.7b

# Best balance of speed and quality
LLM_MODEL_NAME=mistral:latest

# Best quality (for final demo)
LLM_MODEL_NAME=llama3:latest
```

---

## Project Structure

```
multi-agent-system/
├── docker-compose.yml      # Service orchestration
├── .env.example            # Environment template
├── README.md               # This file
├── docs/
│   ├── architecture.md     # System diagrams
│   └── design.md           # Design decisions
├── backend/
│   ├── main.py             # FastAPI application
│   ├── config.py           # Model-agnostic configuration
│   ├── graph/              # LangGraph state machine
│   ├── tools/              # Tool adapters (RAG, search, Reddit)
│   ├── memory/             # Short and long-term memory
│   └── api/                # HTTP endpoints
├── frontend/
│   └── index.html          # Single-file UI
└── scripts/
    └── ingest_pubmed.py    # Knowledge base ingestion
```

---

## Design Notes

See [docs/design.md](docs/design.md) for detailed explanation of:
- Orchestration decisions (why LangGraph, why Planner→Executor→Verifier)
- Memory architecture (short-term vs long-term, SQLite choice)
- Extensibility (adding tools, switching models, swapping vector DB)
- Documented assumptions (Reddit API status, abstracts vs full text)

---

## Known Limitations

- **Response time:** 2-10 minutes with local LLM (by design — Ollama runs on CPU)
- **Reddit API:** Pending approval under Reddit's new policy (2023). Tool degrades gracefully.
- **PubMed:** Abstracts only (full text behind publisher paywalls)
- **No authentication:** Single-user system as per assignment scope

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | LangGraph 0.2.55 | State machine, human-in-the-loop |
| AI Framework | LangChain 0.3.13 | Tool abstractions, model interfaces |
| Backend API | FastAPI 0.115.6 | REST endpoints |
| Vector DB | Chroma 0.5.23 | Embedding storage and retrieval |
| Local LLM | Ollama + llama3 | Language model inference |
| Embeddings | nomic-embed-text | Text-to-vector conversion |
| Search | Tavily API | Live web search |
| Reddit | PRAW 7.8.1 | Community intelligence |
| PubMed | NCBI E-utilities | Knowledge base source |
| Containerization | Docker Compose | Multi-service orchestration |
| Memory | SQLite | Long-term memory + checkpoints |