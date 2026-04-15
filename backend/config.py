"""
config.py - Central Configuration & Model Factory
==================================================

PURPOSE:
    Single source of truth for all configuration.
    Implements model-agnostic design by abstracting LLM/embedding
    creation behind factory functions.

THE CORE PATTERN (why this file exists):

    WITHOUT this file (bad - hard-coded):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3:latest")  # change model = change code

    WITH this file (good - configurable):
        from config import get_llm
        llm = get_llm()  # reads .env, creates right model, returns it

    Now to switch from llama3 to mistral, you change ONE line in .env.
    Zero code changes required anywhere else.

HOW pydantic-settings WORKS:
    Settings(BaseSettings) reads your .env file automatically.
    Each class attribute maps to an environment variable by name.

    If .env contains:  LLM_MODEL_NAME=llama3:latest
    Then:              settings.llm_model_name == "llama3:latest"

    pydantic validates types: if you accidentally write LLM_PORT=abc
    and the attribute is typed as int, pydantic raises an error immediately.
    Much better than discovering the error when the port is actually used.

CHANGES FROM PREVIOUS VERSION:
    - Added NCBI/PubMed settings (ncbi_api_key, pubmed_topics,
      pubmed_articles_per_topic)
    - Fixed model name matching in validate_config() to handle
      "nomic-embed-text" vs "nomic-embed-text:latest" tag mismatch
    - Fixed .env path resolution using Path(__file__) so script
      works regardless of which directory you run Python from
"""

import os
from functools import lru_cache
from pathlib import Path

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
    - Converts strings to correct Python types (str->int, str->bool, etc.)

    Field() provides:
    - default: value used if env var is missing
    - description: documentation for what the variable does
    """

    model_config = SettingsConfigDict(
        # Look for .env in the project ROOT folder.
        # Path(__file__) = this file (config.py)
        # .parent        = backend/ folder
        # .parent.parent = multi-agent-system/ (root) folder
        # So .env is always found regardless of where you run Python from.
        # CHANGED: was env_file=".env" which broke when running from
        # inside the backend/ folder. Now uses absolute path.
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----------------------------------------------------------
    # LLM Settings
    # ----------------------------------------------------------

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
            "Range: 0.0 (fully deterministic) to 1.0 (very random). "
            "We use 0.1 for agents - low creativity, high consistency."
        )
    )

    # ----------------------------------------------------------
    # Embedding Model Settings
    # ----------------------------------------------------------

    embedding_provider: str = Field(
        default="ollama",
        description="Which embedding service to use. Options: 'ollama', 'openai'"
    )

    embedding_model_name: str = Field(
        default="nomic-embed-text",
        description=(
            "The embedding model name. "
            "'nomic-embed-text' is our local Ollama embedding model. "
            "It converts text into 768-dimensional vectors for RAG search."
        )
    )

    embedding_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for embedding model. Same logic as llm_base_url."
    )

    # ----------------------------------------------------------
    # API Keys
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # *** NEW SECTION: NCBI / PubMed Settings ***
    # Added in Phase 2 to support PubMed knowledge base ingestion.
    #
    # WHY THESE SETTINGS EXIST:
    # The RAG pipeline needs a knowledge base to search.
    # We use PubMed (free medical research database) as our source.
    # These settings control which topics to fetch and how many articles.
    #
    # The NCBI API key is free and increases rate limit from
    # 3 requests/second to 10 requests/second.
    # Get yours at: https://www.ncbi.nlm.nih.gov/account/
    # ----------------------------------------------------------

    ncbi_api_key: str = Field(
        default="",
        description=(
            "NCBI API key for PubMed access. "
            "Free at ncbi.nlm.nih.gov/account. "
            "Without key: 3 requests/sec limit. "
            "With key: 10 requests/sec limit. "
            "Speeds up the document ingestion script significantly."
        )
    )

    pubmed_topics: str = Field(
        default="machine learning artificial intelligence,neurological disorders",
        description=(
            "Comma-separated list of topics to search and ingest from PubMed. "
            "Each topic is searched separately and stored with its own "
            "topic metadata tag in Chroma, enabling topic-specific filtering. "
            "Example: 'diabetes treatment,cancer immunotherapy,AI in radiology'"
        )
    )

    pubmed_articles_per_topic: int = Field(
        default=100,
        description=(
            "How many articles to fetch per topic during ingestion. "
            "More articles = better RAG coverage but slower ingestion. "
            "Recommended: 50 for testing, 100 for demo, 200 for production."
        )
    )

    # ----------------------------------------------------------
    # Chroma Vector Database
    # ----------------------------------------------------------

    chroma_host: str = Field(
        default="localhost",
        description=(
            "Chroma DB host. "
            "Use 'localhost' for direct Python execution. "
            "Use 'chroma' (the Docker service name) inside Docker containers."
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
            "A collection is like a table in a relational database. "
            "All topics share one collection, differentiated by metadata."
        )
    )

    # ----------------------------------------------------------
    # Application Settings
    # ----------------------------------------------------------

    backend_host: str = Field(
        default="0.0.0.0",
        description=(
            "Host to bind the FastAPI server to. "
            "0.0.0.0 = accept connections from any network interface. "
            "Required for Docker - containers need this to accept external connections."
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

    # ----------------------------------------------------------
    # Memory Settings
    # ----------------------------------------------------------

    long_term_memory_path: str = Field(
        default="./memory_store.db",
        description=(
            "File path for the SQLite database used for long-term memory. "
            "SQLite stores everything in a single .db file - no server needed."
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

    @lru_cache() means this function only executes ONCE.
    Every subsequent call returns the same cached Settings object.

    WHY CACHE?
    Without caching, every call to get_settings() would:
    1. Open and read the .env file from disk
    2. Parse every line
    3. Validate all values with pydantic
    4. Create a new Settings object

    This happens many times per request (every tool, every node, every agent).
    Caching means we do all that work exactly ONCE per application run.

    USAGE EXAMPLE:
        from config import get_settings
        settings = get_settings()
        print(settings.llm_model_name)              # "llama3:latest"
        print(settings.pubmed_articles_per_topic)   # 100
        print(settings.ncbi_api_key)                # "your_key_here"
    """
    return Settings()


# ================================================================
# LLM FACTORY FUNCTION
# ================================================================

def get_llm():
    """
    Creates and returns the correct LangChain LLM object based on config.

    THIS IS THE KEY TO MODEL-AGNOSTIC DESIGN.

    Every agent, node, and tool calls get_llm() to get an LLM.
    None of them import ChatOllama or ChatOpenAI directly.
    They just call get_llm() and get back a working model.

    HOW TO SWITCH MODELS:
        Change in .env:  LLM_MODEL_NAME=mistral:latest
        Result: entire system uses Mistral. Zero code changes anywhere.

    HOW TO SWITCH PROVIDERS:
        Change in .env:  LLM_PROVIDER=openai
                         LLM_MODEL_NAME=gpt-4o
        Result: entire system uses GPT-4o. Zero code changes anywhere.

    WHY IMPORT INSIDE THE IF BLOCKS?
        This is called lazy importing. We only import the library
        for the provider that is actually configured.
        If you use Ollama, langchain_openai never gets imported.
        This means you do not need cloud provider packages installed
        if you are only using local models.

    RETURNS:
        A LangChain BaseChatModel instance.
        All LangChain models share the same interface:
            llm.invoke(messages)  -> AIMessage
            llm.stream(messages)  -> streamed tokens
        So all downstream code works identically regardless of provider.
    """
    settings = get_settings()

    # -- Ollama (Local Models) ----------------------------------
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        print(
            f"[Config] LLM: Ollama | "
            f"Model: {settings.llm_model_name} | "
            f"Temp: {settings.llm_temperature}"
        )

        return ChatOllama(
            model=settings.llm_model_name,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            # num_predict=-1 means let the model decide when to stop.
            # Without this, some Ollama models cut off responses early.
            num_predict=-1,
        )

    # -- OpenAI (Cloud) ----------------------------------------
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty but LLM_PROVIDER=openai. "
                "Add your OpenAI API key to .env"
            )

        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        print(
            f"[Config] LLM: OpenAI | "
            f"Model: {settings.llm_model_name} | "
            f"Temp: {settings.llm_temperature}"
        )

        return ChatOpenAI(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )

    # -- Anthropic (Cloud) -------------------------------------
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is empty but LLM_PROVIDER=anthropic. "
                "Add your Anthropic API key to .env"
            )

        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

        print(
            f"[Config] LLM: Anthropic | "
            f"Model: {settings.llm_model_name} | "
            f"Temp: {settings.llm_temperature}"
        )

        return ChatAnthropic(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )

    # -- Unknown Provider --------------------------------------
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
    Creates and returns the correct embedding model based on config.

    IMPORTANT - Embeddings are NOT the same as the LLM:

        LLM (get_llm()):
            Input:  text (a question or instruction)
            Output: text (an answer or response)
            Used by: all agents for reasoning and answering

        Embedding model (get_embeddings()):
            Input:  text (a document chunk or search query)
            Output: a list of 768 numbers (called a vector)
            Used by: RAG pipeline ONLY for document similarity search

    HOW VECTOR SIMILARITY WORKS:
        "Python programming"  -> [0.2,  0.8, -0.1,  0.5, ...]
        "coding in Python"    -> [0.21, 0.79, -0.09, 0.51, ...]  similar!
        "cooking pasta"       -> [0.9, -0.3,  0.7, -0.2, ...]    different

        Chroma stores these vectors and finds the closest ones to your query.
        Closeness is measured using cosine similarity (angle between vectors).

    RETURNS:
        A LangChain Embeddings instance with methods:
            .embed_documents(texts) -> list of vectors
            .embed_query(text)      -> single vector
        Chroma calls these methods internally.
    """
    settings = get_settings()

    # -- Ollama Embeddings (Local) ------------------------------
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        print(
            f"[Config] Embeddings: Ollama | "
            f"Model: {settings.embedding_model_name}"
        )

        return OllamaEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.embedding_base_url,
        )

    # -- OpenAI Embeddings (Cloud) -----------------------------
    elif settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty but EMBEDDING_PROVIDER=openai. "
                "Add your OpenAI API key to .env"
            )

        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        print(
            f"[Config] Embeddings: OpenAI | "
            f"Model: {settings.embedding_model_name}"
        )

        return OpenAIEmbeddings(
            model=settings.embedding_model_name or "text-embedding-3-small",
        )

    # -- Unknown Provider --------------------------------------
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
    Validates the configuration at startup and prints a status report.

    Called once when the application starts to catch problems early,
    before they cause confusing errors deep inside agent code.

    WHY VALIDATE EARLY?
        Without this, a missing API key would only be discovered when
        the Tavily tool is actually called mid-conversation giving
        a confusing error at a random point. Validating at startup
        gives a clear, immediate error with an actionable message.

    *** CHANGED FROM PREVIOUS VERSION ***
        Fixed model name matching to handle Ollama tag format.
        Ollama stores models as "nomic-embed-text:latest" but config
        stores "nomic-embed-text" (without :latest tag).
        Previous version did exact string match - always failed.
        New version uses model_exists() helper that handles both formats.

    RETURNS:
        dict with keys:
            'valid'    : bool  - True if no errors found
            'warnings' : list  - Non-fatal issues (missing optional keys)
            'errors'   : list  - Fatal issues (Ollama not running, etc.)
    """
    settings = get_settings()
    warnings = []
    errors = []

    # -- Check Ollama connectivity and model availability ------
    if settings.llm_provider == "ollama":
        import httpx
        try:
            response = httpx.get(
                f"{settings.llm_base_url}/api/tags",
                timeout=5.0
            )

            if response.status_code == 200:
                # Ollama returns model names like "llama3:latest"
                # Our config might store "llama3" without the :latest tag.
                # We need flexible matching that handles both formats.
                models = [
                    m["name"]
                    for m in response.json().get("models", [])
                ]

                # *** CHANGED: new helper function for flexible matching ***
                def model_exists(model_name: str, available: list) -> bool:
                    """
                    Check if model_name matches any model in available list.

                    Handles two cases:
                    1. Exact match:   "llama3:latest" == "llama3:latest"
                    2. Tag mismatch:  "llama3" matches "llama3:latest"
                                      splits on ":" and compares base name

                    This is necessary because Ollama always appends :latest
                    or another tag to model names in its list, but users
                    and configs often omit the tag.
                    """
                    return any(
                        model_name == m or              # exact match
                        model_name == m.split(":")[0]   # base name match
                        for m in available
                    )

                # Check LLM model is available in Ollama
                if not model_exists(settings.llm_model_name, models):
                    errors.append(
                        f"LLM model '{settings.llm_model_name}' not found. "
                        f"Available: {models}. "
                        f"Run: ollama pull {settings.llm_model_name}"
                    )

                # Check embedding model is available in Ollama
                if not model_exists(settings.embedding_model_name, models):
                    errors.append(
                        f"Embedding model '{settings.embedding_model_name}' "
                        f"not found. Available: {models}. "
                        f"Run: ollama pull {settings.embedding_model_name}"
                    )

            else:
                errors.append(
                    f"Ollama returned unexpected status {response.status_code}."
                )

        except httpx.ConnectError:
            errors.append(
                f"Cannot connect to Ollama at {settings.llm_base_url}. "
                f"Is Ollama running? Open the Ollama app or run 'ollama serve'."
            )

        except httpx.TimeoutException:
            warnings.append(
                f"Ollama at {settings.llm_base_url} is slow to respond. "
                f"It may be loading a model."
            )

    # -- Check Tavily API key ----------------------------------
    if not settings.tavily_api_key:
        warnings.append(
            "TAVILY_API_KEY is empty. Internet search will not work. "
            "Get a free key at tavily.com"
        )

    # -- Check Reddit credentials ------------------------------
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        warnings.append(
            "Reddit credentials missing. Reddit intelligence will not work. "
            "Create an app at reddit.com/prefs/apps"
        )

    # *** NEW: Check NCBI API key (warning only - not required) ***
    if not settings.ncbi_api_key:
        warnings.append(
            "NCBI_API_KEY is empty. PubMed ingestion will be rate-limited "
            "to 3 requests/second instead of 10. "
            "Get a free key at ncbi.nlm.nih.gov/account"
        )

    # -- Build and print the report ----------------------------
    valid = len(errors) == 0

    print("\n" + "=" * 50)
    print("Configuration Validation Report")
    print("=" * 50)
    print(f"LLM Provider:        {settings.llm_provider}")
    print(f"LLM Model:           {settings.llm_model_name}")
    print(f"Embedding Provider:  {settings.embedding_provider}")
    print(f"Embedding Model:     {settings.embedding_model_name}")
    print(f"Chroma:              {settings.chroma_host}:{settings.chroma_port}")
    print(f"Tavily Key:          {'OK - set' if settings.tavily_api_key else 'MISSING'}")
    print(f"Reddit Credentials:  {'OK - set' if settings.reddit_client_id else 'MISSING'}")
    # *** NEW: show NCBI key status in report ***
    print(f"NCBI API Key:        {'OK - set' if settings.ncbi_api_key else 'MISSING (optional)'}")
    print(f"PubMed Topics:       {settings.pubmed_topics}")
    print(f"Articles per topic:  {settings.pubmed_articles_per_topic}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [!] {w}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  [X] {e}")

    print(f"\nStatus: {'VALID' if valid else 'INVALID - fix errors above'}")
    print("=" * 50 + "\n")

    return {"valid": valid, "warnings": warnings, "errors": errors}