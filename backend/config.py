"""
config.py — Central Configuration & Model Factory
==================================================

PURPOSE:
    Single source of truth for all configuration.
    Implements model-agnostic design by abstracting LLM/embedding
    creation behind factory functions.

THE CORE PATTERN (why this file exists):
    
    WITHOUT this file (bad — hard-coded):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3:latest")  # ← change model = change code
    
    WITH this file (good — configurable):
        from config import get_llm
        llm = get_llm()  # ← reads .env, creates right model, returns it
    
    Now to switch from llama3 to mistral, you change ONE character in .env.
    Zero code changes required anywhere.

HOW pydantic-settings WORKS:
    Settings(BaseSettings) reads your .env file automatically.
    Each class attribute maps to an environment variable by name.
    
    If .env contains:  LLM_MODEL_NAME=llama3:latest
    Then:              settings.llm_model_name == "llama3:latest"
    
    pydantic validates types: if you accidentally write LLM_PORT=abc
    and the attribute is typed as int, pydantic raises an error immediately.
    Much better than discovering the error when the port is actually used.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ================================================================
# SETTINGS CLASS
# ================================================================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    
    BaseSettings from pydantic-settings:
    - Automatically reads .env file
    - Validates all values on instantiation  
    - Raises clear errors for missing required values
    - Converts strings to correct Python types (str→int, str→bool, etc.)
    
    Field() provides:
    - default: value used if env var is missing
    - description: documentation for what the variable does
    """
    
    # SettingsConfigDict tells pydantic-settings how to behave
    model_config = SettingsConfigDict(
        # Where to find the .env file
        # ".env" means look in the current working directory
        env_file=".env",
        
        # File encoding for the .env file
        env_file_encoding="utf-8",
        
        # Allow LLM_MODEL_NAME and llm_model_name to both work
        case_sensitive=False,
        
        # Don't crash if .env has variables we don't define here
        extra="ignore",
    )
    
    # ──────────────────────────────────────────────────────────
    # LLM Settings
    # ──────────────────────────────────────────────────────────
    
    llm_provider: str = Field(
        default="ollama",
        description=(
            "Which LLM service to use. "
            "Options: 'ollama' (local), 'openai' (cloud), 'anthropic' (cloud)"
        )
    )
    
    llm_model_name: str = Field(
        default="llama3:latest",
        description=(
            "The specific model to use within the provider. "
            "Ollama options: 'llama3:latest', 'mistral:latest', 'qwen3:1.7b'"
        )
    )
    
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Base URL for the Ollama API. "
            "Use http://localhost:11434 when running Python directly. "
            "Use http://host.docker.internal:11434 when running inside Docker."
        )
    )
    
    llm_temperature: float = Field(
        default=0.1,
        description=(
            "Controls randomness in LLM responses. "
            "Range: 0.0 (deterministic) to 1.0 (very random). "
            "We use 0.1 for agents — low creativity, high consistency. "
            "Agents need to follow instructions, not be creative."
        )
    )
    
    # ──────────────────────────────────────────────────────────
    # Embedding Model Settings
    # ──────────────────────────────────────────────────────────
    
    embedding_provider: str = Field(
        default="ollama",
        description="Which embedding service to use. Options: 'ollama', 'openai'"
    )
    
    embedding_model_name: str = Field(
        default="nomic-embed-text",
        description=(
            "The embedding model name. "
            "'nomic-embed-text' is our local Ollama embedding model. "
            "It converts text to 768-dimensional vectors."
        )
    )
    
    embedding_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for embedding model. Same logic as llm_base_url."
    )
    
    # ──────────────────────────────────────────────────────────
    # API Keys
    # ──────────────────────────────────────────────────────────
    
    tavily_api_key: str = Field(
        default="",
        description="Tavily search API key. Free tier at tavily.com"
    )
    
    reddit_client_id: str = Field(
        default="",
        description="Reddit app client ID from reddit.com/prefs/apps"
    )
    
    reddit_client_secret: str = Field(
        default="",
        description="Reddit app client secret"
    )
    
    reddit_user_agent: str = Field(
        default="multi-agent-system/1.0",
        description=(
            "Required by Reddit API to identify your application. "
            "Format: 'appname/version by u/YourRedditUsername'"
        )
    )
    
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Only needed if llm_provider='openai'"
    )
    
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Only needed if llm_provider='anthropic'"
    )
    
    # ──────────────────────────────────────────────────────────
    # Chroma Vector Database
    # ──────────────────────────────────────────────────────────
    
    chroma_host: str = Field(
        default="localhost",
        description=(
            "Chroma DB host. "
            "Use 'localhost' for direct Python execution. "
            "Use 'chroma' (Docker service name) inside Docker containers."
        )
    )
    
    chroma_port: int = Field(
        default=8001,
        description="Chroma DB HTTP port"
    )
    
    chroma_collection_name: str = Field(
        default="knowledge_base",
        description=(
            "Name of the Chroma collection where documents are stored. "
            "A collection is like a table in a relational database."
        )
    )
    
    # ──────────────────────────────────────────────────────────
    # Application Settings
    # ──────────────────────────────────────────────────────────
    
    backend_host: str = Field(
        default="0.0.0.0",
        description=(
            "Host to bind the FastAPI server to. "
            "0.0.0.0 = accept connections from any network interface. "
            "Required for Docker — containers need this to accept external connections."
        )
    )
    
    backend_port: int = Field(
        default=8000,
        description="Port for the FastAPI backend server"
    )
    
    frontend_port: int = Field(
        default=3000,
        description="Port for the frontend service"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity. Options: DEBUG, INFO, WARNING, ERROR"
    )
    
    # ──────────────────────────────────────────────────────────
    # Memory Settings
    # ──────────────────────────────────────────────────────────
    
    long_term_memory_path: str = Field(
        default="./memory_store.db",
        description=(
            "File path for the SQLite database used for long-term memory. "
            "SQLite stores everything in a single .db file — no server needed."
        )
    )
    
    memory_lookback_count: int = Field(
        default=5,
        description=(
            "How many past conversation summaries to load for context. "
            "When answering a new question, the agent sees summaries of "
            "the last N conversations to provide continuity."
        )
    )


# ================================================================
# SETTINGS SINGLETON
# ================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a CACHED Settings instance.
    
    @lru_cache() (Least Recently Used Cache) means:
    - First call: creates Settings(), parses .env, validates, stores result
    - ALL subsequent calls: returns the stored result immediately
    
    WHY CACHE?
    Without caching, every call to get_settings() would:
    1. Open and read the .env file from disk
    2. Parse each line  
    3. Validate all values with pydantic
    4. Create a new Settings object
    
    This happens MANY times per request (every tool, every node, etc.)
    Caching means we do it ONCE ever. Much more efficient.
    
    USAGE EXAMPLE:
        from config import get_settings
        settings = get_settings()
        print(settings.llm_model_name)    # → "llama3:latest"
        print(settings.chroma_port)       # → 8001
    """
    return Settings()


# ================================================================
# LLM FACTORY FUNCTION
# ================================================================

def get_llm():
    """
    Creates and returns the correct LLM based on configuration.
    
    This is the KEY function for model-agnostic design.
    It reads LLM_PROVIDER and LLM_MODEL_NAME from .env
    and returns the appropriate LangChain LLM object.
    
    ALL agents, nodes, and tools use this function.
    NOT ONE of them imports ChatOllama or ChatOpenAI directly.
    
    SWITCHING MODELS:
        Change .env: LLM_MODEL_NAME=mistral:latest
        Result: entire system uses mistral. Zero code changes.
    
    SWITCHING PROVIDERS:
        Change .env: LLM_PROVIDER=openai + LLM_MODEL_NAME=gpt-4o
        Result: entire system uses GPT-4o. Zero code changes.
    
    WHY IMPORT INSIDE THE FUNCTION?
    We import langchain_ollama/langchain_openai INSIDE each branch,
    not at the top of the file. This is called "lazy importing."
    
    Benefit: if someone uses LLM_PROVIDER=openai, they don't need
    langchain_ollama installed. The import only happens for the
    provider that's actually configured.
    
    RETURNS:
        A LangChain BaseChatModel instance.
        All LangChain chat models have the same interface:
        - llm.invoke(messages) → response
        - llm.stream(messages) → streamed response
        So the rest of the code works identically regardless of provider.
    """
    settings = get_settings()
    
    # ── Ollama (Local Models) ──────────────────────────────────
    if settings.llm_provider == "ollama":
        # Import only when this provider is selected
        from langchain_ollama import ChatOllama
        
        print(
            f"[Config] LLM: Ollama | "
            f"Model: {settings.llm_model_name} | "
            f"URL: {settings.llm_base_url} | "
            f"Temp: {settings.llm_temperature}"
        )
        
        return ChatOllama(
            model=settings.llm_model_name,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            # num_predict controls max output length
            # -1 means unlimited (model decides when to stop)
            num_predict=-1,
        )
    
    # ── OpenAI (Cloud) ────────────────────────────────────────
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        
        # Set the API key as an environment variable
        # langchain_openai reads it from os.environ["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty but LLM_PROVIDER=openai. "
                "Please add your OpenAI API key to .env"
            )
        
        print(
            f"[Config] LLM: OpenAI | "
            f"Model: {settings.llm_model_name} | "
            f"Temp: {settings.llm_temperature}"
        )
        
        return ChatOpenAI(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )
    
    # ── Anthropic (Cloud) ─────────────────────────────────────
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is empty but LLM_PROVIDER=anthropic. "
                "Please add your Anthropic API key to .env"
            )
        
        print(
            f"[Config] LLM: Anthropic | "
            f"Model: {settings.llm_model_name} | "
            f"Temp: {settings.llm_temperature}"
        )
        
        return ChatAnthropic(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )
    
    # ── Unknown Provider ──────────────────────────────────────
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{settings.llm_provider}'\n"
            f"Valid options: 'ollama', 'openai', 'anthropic'\n"
            f"Check your .env file."
        )


# ================================================================
# EMBEDDING FACTORY FUNCTION
# ================================================================

def get_embeddings():
    """
    Creates and returns the correct embedding model based on configuration.
    
    Same model-agnostic pattern as get_llm().
    
    IMPORTANT: Embeddings ≠ LLM
    ─────────────────────────────
    LLM (get_llm()):
        Input: text (a question)
        Output: text (an answer)
        Used by: all agents for reasoning and answering
    
    Embedding model (get_embeddings()):  
        Input: text (a document chunk or query)
        Output: list of ~768 numbers (a vector)
        Used by: RAG pipeline ONLY, for document similarity search
    
    They're completely separate tools for completely different jobs.
    
    HOW EMBEDDING SIMILARITY WORKS:
    ────────────────────────────────
    "Python programming"  → [0.2, 0.8, -0.1, 0.5, ...]
    "coding in Python"    → [0.21, 0.79, -0.09, 0.51, ...] ← very similar!
    "cooking pasta"       → [0.9, -0.3, 0.7, -0.2, ...]    ← very different
    
    We measure similarity using cosine similarity (geometric angle between vectors).
    Small angle = similar meaning. Large angle = different meaning.
    
    RETURNS:
        A LangChain Embeddings instance.
        Has .embed_documents(texts) and .embed_query(text) methods.
        Chroma uses these internally to create and search embeddings.
    """
    settings = get_settings()
    
    # ── Ollama Embeddings (Local) ──────────────────────────────
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        
        print(
            f"[Config] Embeddings: Ollama | "
            f"Model: {settings.embedding_model_name} | "
            f"URL: {settings.embedding_base_url}"
        )
        
        return OllamaEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.embedding_base_url,
        )
    
    # ── OpenAI Embeddings (Cloud) ─────────────────────────────
    elif settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty but EMBEDDING_PROVIDER=openai."
            )
        
        print(
            f"[Config] Embeddings: OpenAI | "
            f"Model: {settings.embedding_model_name}"
        )
        
        # text-embedding-3-small is OpenAI's best value embedding model
        # 1536 dimensions, very accurate, cheap
        return OpenAIEmbeddings(
            model=settings.embedding_model_name or "text-embedding-3-small",
        )
    
    # ── Unknown Provider ──────────────────────────────────────
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{settings.embedding_provider}'\n"
            f"Valid options: 'ollama', 'openai'\n"
            f"Check your .env file."
        )


# ================================================================
# VALIDATION HELPER
# ================================================================

def validate_config() -> dict:
    """
    Validates the configuration and returns a status report.
    
    Called at application startup to catch configuration problems
    before they cause mysterious errors deep in the code.
    
    Returns a dict with:
    - 'valid': bool (True if everything looks good)
    - 'warnings': list of non-fatal issues
    - 'errors': list of fatal issues
    
    USAGE in main.py:
        from config import validate_config
        status = validate_config()
        if not status['valid']:
            print("Configuration errors:", status['errors'])
            exit(1)
    """
    settings = get_settings()
    warnings = []
    errors = []
    
    # Check that Ollama is reachable (if using Ollama)
    if settings.llm_provider == "ollama":
        import httpx
        try:
            response = httpx.get(
                f"{settings.llm_base_url}/api/tags",
                timeout=5.0  # 5 second timeout
            )
            if response.status_code == 200:
                # Parse the list of available models
                models = [m["name"] for m in response.json().get("models", [])]
                
                # Check if our configured LLM model is available
                if settings.llm_model_name not in models:
                    errors.append(
                        f"LLM model '{settings.llm_model_name}' not found in Ollama. "
                        f"Available models: {models}. "
                        f"Run: ollama pull {settings.llm_model_name}"
                    )
                
                # Check if our embedding model is available
                if settings.embedding_model_name not in models:
                    errors.append(
                        f"Embedding model '{settings.embedding_model_name}' not found in Ollama. "
                        f"Run: ollama pull {settings.embedding_model_name}"
                    )
            else:
                errors.append(f"Ollama returned status {response.status_code}")
                
        except httpx.ConnectError:
            errors.append(
                f"Cannot connect to Ollama at {settings.llm_base_url}. "
                f"Is Ollama running? Start it by running the Ollama application."
            )
        except httpx.TimeoutException:
            warnings.append(
                f"Ollama at {settings.llm_base_url} is slow to respond. "
                f"It might be loading a model."
            )
    
    # Check Tavily key
    if not settings.tavily_api_key:
        warnings.append(
            "TAVILY_API_KEY is empty. Internet search will not work. "
            "Get a free key at tavily.com"
        )
    
    # Check Reddit credentials
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        warnings.append(
            "Reddit credentials missing. Reddit intelligence will not work. "
            "Create an app at reddit.com/prefs/apps"
        )
    
    valid = len(errors) == 0
    
    # Print the report
    print("\n" + "="*50)
    print("Configuration Validation Report")
    print("="*50)
    print(f"LLM Provider:        {settings.llm_provider}")
    print(f"LLM Model:           {settings.llm_model_name}")
    print(f"Embedding Provider:  {settings.embedding_provider}")
    print(f"Embedding Model:     {settings.embedding_model_name}")
    print(f"Chroma:              {settings.chroma_host}:{settings.chroma_port}")
    print(f"Tavily Key:          {'✓ set' if settings.tavily_api_key else '✗ missing'}")
    print(f"Reddit Credentials:  {'✓ set' if settings.reddit_client_id else '✗ missing'}")
    
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  ✗  {e}")
    
    print(f"\nStatus: {'✓ Valid' if valid else '✗ Invalid — fix errors above'}")
    print("="*50 + "\n")
    
    return {"valid": valid, "warnings": warnings, "errors": errors}