"""
retrieval_tool.py - Hybrid RAG Retrieval Tool
===============================================

PURPOSE:
    LangChain @tool that searches the Chroma vector database
    using BOTH semantic similarity AND metadata filters.

WHAT IS HYBRID RETRIEVAL?
    Pure vector search: finds documents semantically similar to query
    Pure metadata filter: finds documents matching exact criteria
    HYBRID: does BOTH simultaneously

    Example of pure vector search:
        "treatments for Parkinson's disease"
        -> finds semantically similar chunks
        -> might return old 2010 articles alongside new 2024 ones

    Example of hybrid search:
        "treatments for Parkinson's disease"
        + filter: pub_date >= "2023-01" AND article_type = "Review"
        -> finds semantically similar chunks
        -> BUT only from review articles published in 2023+

    Hybrid is always better when you have meaningful metadata.
    This is EXACTLY what the assignment requires.

WHY IS THIS A @tool?
    LangChain's @tool decorator transforms a Python function into
    something an AI agent can decide to call.

    Without @tool: just a Python function
    With @tool: agent can say "I will use the retrieval_tool to search
                for information about this topic"

    The agent sees the tool's name and description, decides whether to
    call it, and passes arguments. The @tool decorator handles
    converting the function signature into something the LLM understands.
"""

import logging
from typing import Optional
from langchain.tools import tool
from langchain.schema import Document
import chromadb

# Import our config - gets settings from .env
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_embeddings

logger = logging.getLogger(__name__)


def get_chroma_collection():
    """
    Get the Chroma collection for querying.

    WHY A SEPARATE FUNCTION?
    We don't want to connect to Chroma on every tool call.
    But we also can't connect at import time (Chroma might not be running).
    This function is called only when actually needed.
    """
    settings = get_settings()

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    collection = client.get_collection(settings.chroma_collection_name)
    return collection


@tool
def retrieve_documents(
    query: str,
    topic_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    article_type_filter: Optional[str] = None,
    n_results: int = 5,
) -> str:
    """
    Search the medical research knowledge base for relevant articles.

    Use this tool when you need to find information from research papers
    about medical topics, AI, machine learning, or neurological disorders.

    Args:
        query: The search query describing what information you need.
               Example: "deep learning for brain tumor detection"

        topic_filter: Filter by topic area. Options:
                      "machine_learning_artificial_intelligence"
                      "neurological_disorders"
                      Leave empty to search all topics.

        date_from: Only return articles published after this date.
                   Format: "YYYY-MM" (e.g., "2023-01" for Jan 2023+)
                   Leave empty to search all dates.

        article_type_filter: Filter by article type.
                             Options: "Review", "Clinical Trial",
                             "Meta-Analysis", "Journal Article"
                             Leave empty for all types.

        n_results: Number of results to return (default 5, max 10).

    Returns:
        Formatted string with relevant article excerpts and citations.
    """

    logger.info(
        f"[RAG Tool] Query: '{query}' | "
        f"Topic: {topic_filter} | "
        f"Date from: {date_from} | "
        f"Type: {article_type_filter}"
    )

    try:
        # -- Build metadata filter (WHERE clause) ------------------
        # Chroma uses a MongoDB-like filter syntax
        # Multiple conditions use "$and" operator
        where_conditions = []

        if topic_filter:
            # Clean the topic filter (remove spaces, lowercase)
            clean_topic = topic_filter.lower().replace(" ", "_")
            where_conditions.append({
                "topic": {"$eq": clean_topic}
            })

        if date_from:
            # Filter articles published on or after date_from
            # "$gte" means "greater than or equal to"
            # Works because our dates are in "YYYY-MM" format
            # which sorts lexicographically correctly
            where_conditions.append({
                "pub_date": {"$gte": date_from}
            })

        if article_type_filter:
            where_conditions.append({
                "article_type": {"$eq": article_type_filter}
            })

        # Combine conditions with AND logic
        # If only one condition: use it directly
        # If multiple conditions: wrap in {"$and": [...]}
        if len(where_conditions) == 0:
            where_clause = None
        elif len(where_conditions) == 1:
            where_clause = where_conditions[0]
        else:
            where_clause = {"$and": where_conditions}

        # -- Generate query embedding ------------------------------
        # Convert the user's question to a vector
        # Then Chroma finds stored vectors most similar to this one
        embeddings_model = get_embeddings()
        query_vector = embeddings_model.embed_query(query)

        # -- Query Chroma ------------------------------------------
        collection = get_chroma_collection()

        # Perform the actual vector + metadata search
        query_params = {
            "query_embeddings": [query_vector],
            "n_results": min(n_results, 10),  # cap at 10
            "include": ["documents", "metadatas", "distances"],
        }

        # Only add where clause if we have filters
        # Chroma raises an error if where={} (empty filter)
        if where_clause:
            query_params["where"] = where_clause

        results = collection.query(**query_params)

        # -- Format results for LLM consumption -------------------
        # results structure from Chroma:
        # {
        #   "documents": [["chunk text 1", "chunk text 2", ...]],
        #   "metadatas": [[{metadata1}, {metadata2}, ...]],
        #   "distances": [[0.12, 0.24, ...]]  (lower = more similar)
        # }
        # Note: results are wrapped in extra list (batch dimension)

        documents = results["documents"][0]  # [0] unwraps batch dimension
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            return (
                "No relevant articles found in the knowledge base "
                f"for query: '{query}' with the specified filters. "
                "Try broadening your search or removing filters."
            )

        # Build formatted response string
        formatted_results = []
        formatted_results.append(
            f"Found {len(documents)} relevant research articles:\n"
        )

        seen_pmids = set()  # avoid showing same article twice

        for doc_text, metadata, distance in zip(documents, metadatas, distances):
            pmid = metadata.get("pmid", "unknown")

            # Skip duplicate PMIDs (multiple chunks from same article)
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            # Convert distance to similarity score
            # Cosine distance: 0 = identical, 2 = opposite
            # We convert to 0-100% similarity for readability
            similarity = max(0, (1 - distance / 2) * 100)

            result = (
                f"--- Article ---\n"
                f"Title:    {metadata.get('title', 'N/A')}\n"
                f"Authors:  {metadata.get('authors', 'N/A')}\n"
                f"Journal:  {metadata.get('journal', 'N/A')}\n"
                f"Date:     {metadata.get('pub_date', 'N/A')}\n"
                f"Type:     {metadata.get('article_type', 'N/A')}\n"
                f"PMID:     {pmid}\n"
                f"URL:      {metadata.get('url', 'N/A')}\n"
                f"Match:    {similarity:.1f}%\n"
                f"Excerpt:  {doc_text[:400]}...\n"
            )
            formatted_results.append(result)

        return "\n".join(formatted_results)

    except Exception as e:
        error_msg = f"RAG retrieval failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_article_by_pmid(pmid: str) -> str:
    """
    Retrieve a specific article by its PubMed ID (PMID).

    Use this when you need the full details of a specific article
    that was mentioned in previous search results.

    Args:
        pmid: The PubMed ID of the article (e.g., "38291045")

    Returns:
        Full article details including all available metadata.
    """
    logger.info(f"[RAG Tool] Fetching article by PMID: {pmid}")

    try:
        collection = get_chroma_collection()

        # Get all chunks belonging to this PMID
        results = collection.get(
            where={"pmid": {"$eq": pmid}},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            return f"No article found with PMID: {pmid}"

        # Combine all chunks back into full text
        chunks = list(zip(results["documents"], results["metadatas"]))
        # Sort by chunk_index to reconstruct original order
        chunks.sort(key=lambda x: x[1].get("chunk_index", 0))

        metadata = chunks[0][1]  # metadata is same for all chunks
        full_text = " ".join(chunk[0] for chunk in chunks)

        return (
            f"FULL ARTICLE DETAILS\n"
            f"{'='*40}\n"
            f"Title:    {metadata.get('title', 'N/A')}\n"
            f"Authors:  {metadata.get('authors', 'N/A')}\n"
            f"Journal:  {metadata.get('journal', 'N/A')}\n"
            f"Date:     {metadata.get('pub_date', 'N/A')}\n"
            f"Type:     {metadata.get('article_type', 'N/A')}\n"
            f"Topic:    {metadata.get('topic_display', 'N/A')}\n"
            f"PMID:     {pmid}\n"
            f"URL:      {metadata.get('url', 'N/A')}\n"
            f"{'='*40}\n"
            f"CONTENT:\n{full_text}\n"
        )

    except Exception as e:
        error_msg = f"Failed to retrieve article {pmid}: {str(e)}"
        logger.error(error_msg)
        return error_msg