"""
search_tool.py - Internet Search Tool Adapter
===============================================

PURPOSE:
    Wraps the Tavily search API as a LangChain tool that agents can call.
    Provides real-time internet search to complement RAG retrieval.

WHY DO WE NEED INTERNET SEARCH ALONGSIDE RAG?
    RAG (Retrieval): searches YOUR knowledge base (PubMed articles ingested
                     at a fixed point in time). Good for:
                     - Deep knowledge on specific topics
                     - Academic citations and research
                     - Consistent, well-structured information

    Internet Search: searches the live web right now. Good for:
                     - Breaking news and recent events
                     - Information newer than your last ingestion
                     - General facts not in your knowledge base
                     - Current statistics and data

    TOGETHER: comprehensive answers with both deep knowledge AND
              current information. This is why the assignment requires both.

WHY TAVILY SPECIFICALLY?
    - Designed for AI agents (returns clean text, not HTML soup)
    - Free tier: 1000 searches/month
    - Returns relevance scores
    - Can search specific domains (e.g., only search academic sites)
    - Has native LangChain integration
    - Much simpler than Google Search API (which requires OAuth setup)

ASSIGNMENT REQUIREMENT MET:
    "Internet Search Agent: Must be abstracted as a tool adapter,
     cannot be called directly from UI, must return structured output"

    This file IS the tool adapter. It is only callable by agents
    through LangGraph - never directly from the frontend.
    It returns structured JSON-compatible output.
"""

import logging
from typing import Optional
from langchain.tools import tool
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings

logger = logging.getLogger(__name__)


def get_tavily_client():
    """
    Creates and returns a Tavily search client.

    WHY A SEPARATE FUNCTION?
    We don't import TavilyClient at the module level because:
    1. Import happens at Python startup, before .env is loaded
    2. If TAVILY_API_KEY is missing, we get a confusing error at import
       instead of a helpful error when the tool is actually called
    3. Lazy initialization = faster startup, clearer errors

    TavilyClient automatically reads the TAVILY_API_KEY from
    environment variables OR we pass it explicitly.
    We pass it explicitly for clarity and to support our config system.
    """
    from tavily import TavilyClient

    settings = get_settings()

    if not settings.tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set in .env file. "
            "Get a free key at https://tavily.com"
        )

    return TavilyClient(api_key=settings.tavily_api_key)


@tool
def search_internet(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
) -> str:
    """
    Search the internet for current information on any topic.

    Use this tool when you need:
    - Recent news or events (last days/weeks/months)
    - Information that might not be in the research database
    - Current statistics, prices, or rapidly changing data
    - General knowledge questions not covered by research papers
    - Verification of facts from multiple sources

    Do NOT use this for medical research questions that are better
    answered by the research database (use retrieve_documents instead).

    Args:
        query: What to search for. Be specific for better results.
               Example: "latest FDA approved AI diagnostic tools 2024"

        search_depth: How thorough the search should be.
                      "basic"    = faster, good for simple facts (default)
                      "advanced" = slower, better for complex research
                                   Use when basic results are insufficient.

        max_results: Number of results to return (1-10, default 5).
                     More results = more context but longer processing time.

    Returns:
        Formatted search results with titles, URLs, and content summaries.
        Each result includes a relevance score.
    """

    logger.info(
        f"[Search Tool] Query: '{query}' | "
        f"Depth: {search_depth} | "
        f"Max results: {max_results}"
    )

    try:
        client = get_tavily_client()

        # Call Tavily search API
        # search() parameters:
        #   query: the search string
        #   search_depth: "basic" or "advanced"
        #   max_results: how many results to return
        #   include_answer: Tavily can generate a direct answer (we skip this
        #                   because we want raw results for our LLM to process)
        #   include_raw_content: include full page content (too long, skip)
        #   include_images: we don't need images
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )

        # Tavily response structure:
        # {
        #   "query": "your search query",
        #   "results": [
        #     {
        #       "title": "Article title",
        #       "url": "https://...",
        #       "content": "Snippet of article content...",
        #       "score": 0.87,        # relevance score 0-1
        #       "published_date": "2024-01-15"  # not always present
        #     },
        #     ...
        #   ]
        # }

        results = response.get("results", [])

        if not results:
            return (
                f"No internet search results found for: '{query}'. "
                "Try rephrasing your query or using different keywords."
            )

        # Format results into a clean string for the LLM to read
        # We format as structured text that's easy for LLMs to parse
        formatted_lines = [
            f"Internet Search Results for: '{query}'\n"
            f"Found {len(results)} results:\n"
        ]

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content available")
            score = result.get("score", 0)
            pub_date = result.get("published_date", "Date unknown")

            # Truncate content if too long
            # LLMs have context limits - we don't want one result
            # to consume too much of the available context window
            if len(content) > 500:
                content = content[:500] + "..."

            result_text = (
                f"Result {i}:\n"
                f"  Title:     {title}\n"
                f"  URL:       {url}\n"
                f"  Published: {pub_date}\n"
                f"  Relevance: {score:.2f}\n"
                f"  Summary:   {content}\n"
            )
            formatted_lines.append(result_text)

        return "\n".join(formatted_lines)

    except Exception as e:
        error_msg = f"Internet search failed: {str(e)}"
        logger.error(error_msg)
        # Return error as string (not raise) so agent can handle gracefully
        # If we raised an exception, the whole agent pipeline would crash
        # By returning an error string, the agent can decide what to do next
        return error_msg


@tool
def search_medical_news(query: str, max_results: int = 5) -> str:
    """
    Search specifically for recent medical and health news.

    Use this for finding recent clinical developments, drug approvals,
    medical breakthroughs, or health policy news that would not be
    in the research database yet.

    This is different from retrieve_documents (which searches stored
    PubMed research papers) - this searches live medical news sites.

    Args:
        query: Medical topic to search for.
               Example: "new Alzheimer's treatment approved 2024"

        max_results: Number of results to return (default 5).

    Returns:
        Recent medical news results with sources and dates.
    """

    logger.info(f"[Search Tool] Medical news search: '{query}'")

    try:
        client = get_tavily_client()

        # For medical searches, use advanced depth for better accuracy
        # Also restrict to reputable medical/news domains
        response = client.search(
            query=f"{query} medical health clinical",
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            # include_domains restricts results to specific websites
            # These are all reputable medical/science news sources
            include_domains=[
                "pubmed.ncbi.nlm.nih.gov",
                "nih.gov",
                "who.int",
                "cdc.gov",
                "nejm.org",
                "thelancet.com",
                "nature.com",
                "sciencedaily.com",
                "medscape.com",
                "healthline.com",
            ],
        )

        results = response.get("results", [])

        if not results:
            # Fall back to unrestricted search if domain-restricted returns nothing
            logger.info(
                "[Search Tool] No results from restricted domains, "
                "falling back to general search"
            )
            return search_internet.invoke({
                "query": query,
                "max_results": max_results
            })

        formatted_lines = [
            f"Medical News Search Results for: '{query}'\n"
            f"Sources restricted to reputable medical outlets.\n"
            f"Found {len(results)} results:\n"
        ]

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")[:500]
            pub_date = result.get("published_date", "Date unknown")

            formatted_lines.append(
                f"Result {i}:\n"
                f"  Title:     {title}\n"
                f"  Source:    {url}\n"
                f"  Published: {pub_date}\n"
                f"  Content:   {content}\n"
            )

        return "\n".join(formatted_lines)

    except Exception as e:
        error_msg = f"Medical news search failed: {str(e)}"
        logger.error(error_msg)
        return error_msg