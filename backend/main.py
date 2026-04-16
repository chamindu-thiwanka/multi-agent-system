"""
main.py - FastAPI Application Entry Point
==========================================

PURPOSE:
    Creates the FastAPI application, configures middleware,
    and is the file that uvicorn runs to start the server.

HOW FASTAPI STARTS:
    uvicorn main:app --host 0.0.0.0 --port 8000
    
    This means: "run the 'app' object from 'main.py'"
    uvicorn is the ASGI server that handles HTTP connections.
    FastAPI is the framework that defines what to do with them.

MIDDLEWARE:
    Middleware runs on EVERY request before it reaches your endpoints.
    We use CORSMiddleware to allow the frontend to call the backend.

WHAT IS CORS?
    Browsers have a security feature: by default, JavaScript on
    http://localhost:3000 (frontend) cannot call http://localhost:8000
    (backend) because they're on different "origins" (different ports).
    
    CORS (Cross-Origin Resource Sharing) headers tell the browser:
    "It's OK, I allow requests from localhost:3000"
    
    Without CORS, the browser would block all frontend → backend calls
    with an error: "Access to fetch at '...' has been blocked by CORS policy"

LIFESPAN:
    The @asynccontextmanager lifespan function runs:
    - STARTUP code: before the server accepts any requests
    - SHUTDOWN code: after the server stops accepting requests
    
    We use startup to:
    - Validate configuration (API keys, Ollama running, etc.)
    - Initialize the memory database
    - Pre-load the LangGraph (so first request isn't slow)
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings, validate_config
from api.routes import router
from memory.long_term import initialize_memory_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================================================================
# LIFESPAN MANAGER
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.
    
    Code BEFORE 'yield' runs on startup.
    Code AFTER 'yield' runs on shutdown.
    
    The 'yield' is where FastAPI runs and handles requests.
    Think of it like a try/finally block:
        startup code
        yield  ← server runs here, handling requests
        shutdown code
    """
    
    # ── STARTUP ──────────────────────────────────────────────
    logger.info("="*50)
    logger.info("Multi-Agent System Starting Up")
    logger.info("="*50)
    
    # Validate configuration first
    # This catches missing API keys, Ollama not running, etc.
    # before any requests arrive
    config_status = validate_config()
    if not config_status["valid"]:
        logger.error("Configuration invalid - server starting anyway but some features may fail")
    
    # Initialize the long-term memory SQLite database
    # Creates the table if it doesn't exist
    initialize_memory_db()
    logger.info("Long-term memory database ready")
    
    # Pre-load the LangGraph
    # This compiles the graph and opens the SQLite checkpointer
    # Doing it now means the first /query request is faster
    try:
        from graph.graph import get_graph
        get_graph()
        logger.info("LangGraph pipeline loaded and ready")
    except Exception as e:
        logger.error(f"Failed to pre-load graph: {e}")
    
    settings = get_settings()
    logger.info(f"Server ready on http://{settings.backend_host}:{settings.backend_port}")
    logger.info(f"API docs available at http://localhost:{settings.backend_port}/docs")
    logger.info("="*50)
    
    yield  # Server runs here
    
    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("Multi-Agent System shutting down...")


# ================================================================
# CREATE FASTAPI APPLICATION
# ================================================================

app = FastAPI(
    title="Multi-Agent Research System",
    description=(
        "A LangGraph-orchestrated multi-agent system that answers "
        "complex queries using RAG, internet search, and Reddit intelligence. "
        "Features human-in-the-loop approval before final output."
    ),
    version="1.0.0",
    
    # lifespan handles startup/shutdown
    lifespan=lifespan,
    
    # docs_url: where to find interactive API documentation
    # FastAPI auto-generates this from your endpoint definitions
    # Visit http://localhost:8000/docs to explore and test your API
    docs_url="/docs",
    redoc_url="/redoc",
)


# ================================================================
# CORS MIDDLEWARE
# ================================================================

app.add_middleware(
    CORSMiddleware,
    
    # Which origins (protocol + domain + port) can call this API
    # In development, we allow localhost on common ports
    # In production, you'd replace with your actual domain
    allow_origins=[
        "http://localhost:3000",   # our frontend service
        "http://localhost:8080",   # alternative frontend port
        "http://127.0.0.1:3000",  # same but with IP
        "http://127.0.0.1:8080",
    ],
    
    # Allow cookies and authorization headers to be sent
    allow_credentials=True,
    
    # Which HTTP methods are allowed (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],  # allow all methods
    
    # Which HTTP headers can be sent in requests
    allow_headers=["*"],  # allow all headers
)


# ================================================================
# INCLUDE ROUTES
# ================================================================

# Include all routes from api/routes.py
# prefix="/api/v1" means all routes are prefixed
# So a route defined as "/query" becomes "/api/v1/query"
# This is versioning: if you change the API, create /api/v2/
# Old clients using /api/v1/ still work

app.include_router(router, prefix="/api/v1")


# ================================================================
# HEALTH CHECK ENDPOINT
# ================================================================

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Used by Docker to verify the service is running.
    Returns 200 OK with basic status information.
    
    Also useful for:
    - Load balancers checking if the server is alive
    - Monitoring systems
    - Quick manual verification: curl http://localhost:8000/health
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "multi-agent-system",
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model_name,
    }


@app.get("/")
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "message": "Multi-Agent Research System API",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }


# ================================================================
# RUN DIRECTLY (development only)
# ================================================================

if __name__ == "__main__":
    """
    Allows running with: python main.py
    
    This is for development only.
    In production (Docker), uvicorn is called directly:
        uvicorn main:app --host 0.0.0.0 --port 8000
    
    reload=True means: watch for file changes and restart automatically.
    This is the "hot reload" feature - change a file, server restarts.
    Only use reload=True in development, never in production.
    """
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,           # auto-restart on file changes
        log_level="info",
    )