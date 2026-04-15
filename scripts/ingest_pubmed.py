"""
ingest_pubmed.py - PubMed Knowledge Base Ingestion Script
==========================================================

PURPOSE:
    Fetches research articles from PubMed API and stores them
    in Chroma vector database for RAG retrieval.

HOW TO RUN:
    From the project root folder:
        python scripts/ingest_pubmed.py

    Or with custom options:
        python scripts/ingest_pubmed.py --topics "diabetes,cancer" --limit 50

WHAT THIS SCRIPT DOES (step by step):
    1. Connects to PubMed API (free, no signup required for basic use)
    2. Searches for articles matching each configured topic
    3. Fetches full details (title, abstract, metadata) for each article
    4. Creates LangChain Document objects with rich metadata
    5. Splits long abstracts into smaller chunks for better retrieval
    6. Generates vector embeddings using nomic-embed-text (via Ollama)
    7. Stores everything in Chroma vector database

WHEN TO RUN:
    - Once before starting the system for the first time
    - Whenever you want to update/refresh the knowledge base
    - After changing PUBMED_TOPICS in .env

WHY PUBMED?
    - Free API, no credit card required
    - 35+ million real research articles
    - Rich metadata (date, journal, article type, authors)
    - Demonstrates genuine hybrid retrieval (vector + metadata filters)
    - Structured citations (PMID + URL) satisfy assignment requirements
"""

import sys
import time
import argparse
import logging
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add the backend directory to Python path so we can import config
# sys.path is the list of directories Python searches when you do "import X"
# We need backend/ in there so "from config import..." works
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb

from config import get_settings, get_embeddings

# ================================================================
# LOGGING SETUP
# ================================================================
# Set up logging so we can see what the script is doing.
# This is important for a ~5 minute ingestion process —
# without logging you'd stare at a blank terminal wondering
# if it's working or frozen.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ================================================================
# PUBMED API CLIENT
# ================================================================

class PubMedClient:
    """
    Client for the NCBI E-utilities API (PubMed).

    The NCBI E-utilities API has two main endpoints we use:
    1. esearch: search for articles matching a query
               returns a list of PMIDs (PubMed IDs)
    2. efetch:  fetch full details for specific PMIDs
               returns XML with title, abstract, metadata

    API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25499/

    HOW PUBMED SEARCH WORKS:
        esearch?term=machine+learning&retmax=100
        → returns ["38291045", "38290123", "38289456", ...]  (list of IDs)

        efetch?id=38291045,38290123&rettype=abstract&retmode=xml
        → returns XML with full article details

    WHY XML?
        PubMed's efetch returns richer data in XML format than JSON.
        We parse the XML manually to extract what we need.
    """

    # Base URL for all NCBI E-utilities requests
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: str = ""):
        """
        Initialize the PubMed client.

        api_key: Your free NCBI API key.
                 Without it: limited to 3 requests/second
                 With it: 10 requests/second allowed
        """
        self.api_key = api_key

        # httpx.Client is a synchronous HTTP client
        # timeout=30 means: fail if no response within 30 seconds
        self.client = httpx.Client(timeout=30.0)

        # Rate limiting: how long to wait between requests
        # Without API key: wait 0.4s (safe under 3/sec limit)
        # With API key:    wait 0.15s (safe under 10/sec limit)
        self.request_delay = 0.15 if api_key else 0.4

        if api_key:
            logger.info("PubMed client initialized with API key (10 req/sec)")
        else:
            logger.info("PubMed client initialized without API key (3 req/sec)")

    def _make_request(self, endpoint: str, params: dict) -> httpx.Response:
        """
        Make a rate-limited request to the NCBI API.

        WHAT IS RATE LIMITING?
        APIs limit how many requests you can make per second to prevent
        overloading their servers. If you exceed the limit, they block you.

        We handle this by sleeping between requests.
        time.sleep(0.4) pauses execution for 0.4 seconds.
        At 0.4s delay: max 2.5 requests/sec (safely under 3/sec limit)

        RETRY LOGIC:
        Network requests can fail temporarily (timeout, server busy).
        We retry up to 3 times before giving up.
        This makes the script resilient to brief network issues.
        """
        # Add API key to all requests if we have one
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        # Retry loop: try up to 3 times
        for attempt in range(3):
            try:
                # Wait before making request (rate limiting)
                time.sleep(self.request_delay)

                response = self.client.get(url, params=params)
                response.raise_for_status()  # raises exception if status >= 400
                return response

            except httpx.TimeoutException:
                logger.warning(f"Request timed out (attempt {attempt + 1}/3)")
                if attempt < 2:
                    # Wait longer before retrying
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                else:
                    raise

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # 429 = "Too Many Requests" - we're hitting rate limits
                    logger.warning("Rate limit hit, waiting 5 seconds...")
                    time.sleep(5)
                else:
                    raise

    def search_articles(self, query: str, max_results: int = 100) -> list[str]:
        """
        Search PubMed for articles matching a query.

        RETURNS: List of PMIDs (PubMed IDs) like ["38291045", "38290123"]

        HOW PUBMED SEARCH QUERIES WORK:
        - Simple text: "machine learning" searches all fields
        - Field tags: "machine learning[Title/Abstract]" searches specific fields
        - Date filter: "2020:2024[pdat]" filters by publication date
        - Boolean: "machine learning AND neural networks"

        We add "[Title/Abstract]" to search only title and abstract fields.
        This gives more relevant results than searching all fields
        (which includes author names, journal names, etc.)

        We also filter to only English-language articles with abstracts
        using "hasabstract" and "English[lang]" filters.
        Without abstracts, we have no content for RAG retrieval.
        """
        logger.info(f"Searching PubMed for: '{query}' (max {max_results})")

        params = {
            "db": "pubmed",           # database to search
            "term": (
                f"({query}[Title/Abstract]) "
                f"AND hasabstract "       # must have an abstract
                f"AND English[lang] "     # English only
                f"AND 2019:2024[pdat]"    # published 2019-2024 (recent articles)
            ),
            "retmax": max_results,        # maximum results to return
            "retmode": "json",            # return results as JSON
            "sort": "relevance",          # sort by relevance to query
        }

        response = self._make_request("esearch.fcgi", params)
        data = response.json()

        # The response has this structure:
        # {"esearchresult": {"idlist": ["38291045", "38290123", ...]}}
        pmids = data.get("esearchresult", {}).get("idlist", [])

        logger.info(f"Found {len(pmids)} articles for '{query}'")
        return pmids

    def fetch_article_details(self, pmids: list[str]) -> list[dict]:
        """
        Fetch full details for a list of PMIDs.

        We fetch in BATCHES of 20 to avoid overwhelming the API
        with requests containing hundreds of IDs.

        WHAT WE EXTRACT FROM EACH ARTICLE:
        - pmid:         unique PubMed identifier (our citation ID)
        - title:        article title
        - abstract:     main content for RAG (150-300 words typically)
        - authors:      list of author names
        - journal:      journal name
        - pub_date:     publication date (YYYY-MM format)
        - article_type: "Journal Article", "Review", "Clinical Trial", etc.
        - doi:          Digital Object Identifier (another citation format)
        - url:          direct link to PubMed article page

        WHY IS pub_date IMPORTANT?
        It enables temporal metadata filtering:
        "Find studies from 2023 or later" → filter where pub_date >= "2023"
        This demonstrates hybrid retrieval to evaluators.
        """
        all_articles = []
        batch_size = 20  # fetch 20 articles at a time

        # Split pmids into batches
        # range(0, 100, 20) = [0, 20, 40, 60, 80]
        # So we process: pmids[0:20], pmids[20:40], etc.
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(pmids) + batch_size - 1) // batch_size

            logger.info(
                f"Fetching article details: batch {batch_num}/{total_batches} "
                f"({len(batch)} articles)"
            )

            params = {
                "db": "pubmed",
                "id": ",".join(batch),     # comma-separated list of PMIDs
                "rettype": "abstract",      # return abstract format
                "retmode": "xml",           # return as XML (richer than JSON)
            }

            response = self._make_request("efetch.fcgi", params)

            # Parse the XML response
            articles = self._parse_xml_response(response.text, batch)
            all_articles.extend(articles)

            logger.info(
                f"Successfully parsed {len(articles)}/{len(batch)} articles "
                f"from batch {batch_num}"
            )

        return all_articles

    def _parse_xml_response(
        self, xml_text: str, pmids: list[str]
    ) -> list[dict]:
        """
        Parse PubMed XML response into structured article dictionaries.

        WHY PARSE XML MANUALLY?
        PubMed's XML structure is complex and nested. We use Python's
        built-in xml.etree.ElementTree module to navigate it.

        PUBMED XML STRUCTURE (simplified):
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>38291045</PMID>
              <Article>
                <ArticleTitle>Deep learning for...</ArticleTitle>
                <Abstract>
                  <AbstractText>Background: ...</AbstractText>
                </Abstract>
                <AuthorList>
                  <Author>
                    <LastName>Smith</LastName>
                    <ForeName>John</ForeName>
                  </Author>
                </AuthorList>
                <Journal>
                  <Title>Nature Medicine</Title>
                  <JournalIssue>
                    <PubDate><Year>2024</Year><Month>Feb</Month></PubDate>
                  </JournalIssue>
                </Journal>
              </Article>
              <PublicationTypeList>
                <PublicationType>Review</PublicationType>
              </PublicationTypeList>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        import xml.etree.ElementTree as ET

        articles = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return articles

        # Iterate over each article in the response
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = self._extract_article_data(article_elem)
                if article and article.get("abstract"):
                    # Only include articles that have abstracts
                    # Articles without abstracts are useless for RAG
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse one article: {e}")
                continue

        return articles

    def _extract_article_data(self, article_elem) -> Optional[dict]:
        """
        Extract data from a single PubmedArticle XML element.

        Uses XML XPath expressions to navigate the tree:
        ".//PMID" means "find PMID anywhere in this element's subtree"
        ".//ArticleTitle" means "find ArticleTitle anywhere"

        The .text property gets the text content of an XML element.
        We use "or ''" to default to empty string if element not found.
        """
        import xml.etree.ElementTree as ET

        # Extract PMID (unique identifier)
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is None:
            return None
        pmid = pmid_elem.text

        # Extract title
        title_elem = article_elem.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""
        # itertext() handles cases where title contains nested XML tags
        # e.g. <ArticleTitle>Study of <i>E. coli</i> in...</ArticleTitle>

        # Extract abstract
        # Some articles have structured abstracts with multiple sections
        # We concatenate all sections into one text
        abstract_parts = []
        for abstract_elem in article_elem.findall(".//AbstractText"):
            # Some sections have a Label attribute like "BACKGROUND", "METHODS"
            label = abstract_elem.get("Label", "")
            text = "".join(abstract_elem.itertext())
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Skip articles without meaningful abstracts
        if len(abstract) < 50:
            return None

        # Extract authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            if last:
                # Format: "Smith J" (standard citation format)
                authors.append(f"{last} {first[0] if first else ''}".strip())
        # Limit to first 5 authors (some papers have 100+ authors)
        authors_str = ", ".join(authors[:5])
        if len(authors) > 5:
            authors_str += " et al."

        # Extract journal name
        journal = article_elem.findtext(".//Journal/Title", "")

        # Extract publication date
        # PubMed stores dates in various formats - we normalize to YYYY-MM
        pub_date = self._extract_pub_date(article_elem)

        # Extract article types
        # Examples: "Journal Article", "Review", "Clinical Trial",
        #           "Meta-Analysis", "Randomized Controlled Trial"
        article_types = [
            elem.text
            for elem in article_elem.findall(".//PublicationType")
            if elem.text
        ]
        # Take the most specific type (usually the first non-generic one)
        primary_type = "Journal Article"
        for t in article_types:
            if t not in ["Journal Article", "English Abstract"]:
                primary_type = t
                break

        # Extract DOI for citation
        doi = ""
        for id_elem in article_elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text or ""
                break

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors_str,
            "journal": journal,
            "pub_date": pub_date,
            "article_type": primary_type,
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    def _extract_pub_date(self, article_elem) -> str:
        """
        Extract and normalize publication date to YYYY-MM format.

        WHY NORMALIZE DATES?
        PubMed stores dates in multiple formats:
        - <Year>2024</Year><Month>Feb</Month>
        - <Year>2024</Year><Month>02</Month>
        - <MedlineDate>2024 Jan-Feb</MedlineDate>

        We normalize everything to "YYYY-MM" so metadata filtering
        works consistently:
            where={"pub_date": {"$gte": "2023-01"}}
        """
        # Month name to number mapping
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12"
        }

        # Try structured date first (Year + Month fields)
        year_elem = article_elem.find(".//PubDate/Year")
        month_elem = article_elem.find(".//PubDate/Month")

        if year_elem is not None and year_elem.text:
            year = year_elem.text

            if month_elem is not None and month_elem.text:
                month = month_elem.text
                # Handle both numeric "02" and text "Feb" month formats
                if month.isdigit():
                    return f"{year}-{month.zfill(2)}"
                else:
                    month_num = month_map.get(month[:3].lower(), "01")
                    return f"{year}-{month_num}"
            else:
                return f"{year}-01"

        # Fall back to MedlineDate (e.g. "2024 Jan-Feb")
        medline_elem = article_elem.find(".//MedlineDate")
        if medline_elem is not None and medline_elem.text:
            parts = medline_elem.text.split()
            if parts:
                year = parts[0]
                month = "01"
                if len(parts) > 1:
                    month_str = parts[1][:3].lower()
                    month = month_map.get(month_str, "01")
                return f"{year}-{month}"

        return "2000-01"  # fallback for articles with no date info


# ================================================================
# CHROMA STORE MANAGER
# ================================================================

class ChromaStoreManager:
    """
    Manages the Chroma vector database connection and operations.

    WHAT IS A CHROMA COLLECTION?
    A collection is like a table in a relational database.
    It stores related documents together.
    We use ONE collection called "knowledge_base" for all our articles.

    WHY ONE COLLECTION?
    We differentiate topics using METADATA (the "topic" field).
    This lets us either:
    - Search everything: no filter
    - Search one topic: where={"topic": "machine_learning_ai"}
    - Search by date:   where={"pub_date": {"$gte": "2023-01"}}

    CHROMA CLIENT TYPES:
    - HttpClient: connects to Chroma running as a separate service (Docker)
                  This is what we use here.
    - PersistentClient: connects to Chroma running as a local file
                        Simpler but doesn't work across services.

    We use HttpClient because our Chroma runs in Docker on port 8001.
    """

    def __init__(self):
        settings = get_settings()

        logger.info(
            f"Connecting to Chroma at "
            f"{settings.chroma_host}:{settings.chroma_port}"
        )

        # Connect to Chroma HTTP server (running in Docker)
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self.collection_name = settings.chroma_collection_name

    def get_or_create_collection(self):
        """
        Get the knowledge_base collection, creating it if it doesn't exist.

        get_or_create_collection is idempotent:
        - First run: creates the collection
        - Subsequent runs: returns the existing collection

        DISTANCE FUNCTION:
        We use "cosine" distance (cosine similarity).
        This measures the ANGLE between two vectors.
        Small angle = similar meaning. Large angle = different meaning.

        Why cosine instead of L2 (Euclidean distance)?
        Cosine is better for text because it measures direction, not magnitude.
        "AI" and "Artificial Intelligence AI AI AI AI" would have very
        different L2 distances but similar cosine similarity.
        """
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                # Use cosine similarity for text comparison
                "hnsw:space": "cosine"
            }
        )
        logger.info(
            f"Using collection '{self.collection_name}' "
            f"(currently has {collection.count()} documents)"
        )
        return collection

    def collection_has_topic(self, collection, topic_key: str) -> bool:
        """
        Check if a topic has already been ingested.

        WHY CHECK?
        If you run the ingestion script twice, you don't want duplicate
        articles in Chroma. We check if any documents exist with
        metadata topic=topic_key before ingesting.

        Chroma's where filter syntax:
        {"topic": {"$eq": "machine_learning_ai"}}
        This means: find documents where topic equals "machine_learning_ai"
        """
        try:
            results = collection.get(
                where={"topic": {"$eq": topic_key}},
                limit=1  # we only need to know if ANY exist
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    def delete_topic(self, collection, topic_key: str):
        """
        Delete all documents for a topic (for re-ingestion).
        Called when user explicitly wants to refresh a topic.
        """
        collection.delete(
            where={"topic": {"$eq": topic_key}}
        )
        logger.info(f"Deleted existing documents for topic '{topic_key}'")


# ================================================================
# DOCUMENT PROCESSOR
# ================================================================

class DocumentProcessor:
    """
    Converts raw PubMed articles into LangChain Documents
    ready for embedding and storage in Chroma.

    WHAT IS A LANGCHAIN DOCUMENT?
    LangChain's Document class has two fields:
        page_content: str    - the actual text to embed and search
        metadata: dict       - structured data about this chunk

    WHY SEPARATE CONTENT FROM METADATA?
    - page_content gets embedded (converted to vector)
    - metadata is stored separately and used for filtering
    - You can filter by metadata WITHOUT semantic search
    - You can do semantic search AND filter by metadata simultaneously

    EXAMPLE:
        Document(
            page_content="This study examines deep learning approaches
                          for early detection of Alzheimer's disease...",
            metadata={
                "pmid": "38291045",
                "title": "Deep learning for Alzheimer's detection",
                "journal": "Nature Neuroscience",
                "pub_date": "2024-02",
                "article_type": "Review",
                "topic": "neurological_disorders",
                "source": "pubmed",
                "url": "https://pubmed.ncbi.nlm.nih.gov/38291045/"
            }
        )
    """

    def __init__(self):
        # RecursiveCharacterTextSplitter splits text into chunks.
        #
        # chunk_size=500: each chunk is at most 500 characters
        # WHY 500? PubMed abstracts are 150-300 words (~900-1800 chars).
        # 500 chars ~ 80-100 words. This gives us 2-4 chunks per abstract.
        # Smaller chunks = more precise retrieval but more chunks to search.
        # 500 chars is a good balance for abstracts.
        #
        # chunk_overlap=50: consecutive chunks share 50 characters
        # WHY OVERLAP? At chunk boundaries, context is split mid-thought.
        # Overlap ensures both chunks have some surrounding context.
        # Example: "...diabetes treatment. [chunk1 end]
        #           [chunk2 start] treatment showed significant..."
        # With overlap, chunk2 starts a bit before "treatment" so
        # it still has context about what treatment is being discussed.
        #
        # separators: try to split on these in order
        # First tries paragraph breaks, then sentences, then spaces
        # This means we never cut in the middle of a sentence if possible.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def create_documents(
        self,
        articles: list[dict],
        topic_key: str,
        topic_display: str
    ) -> list[Document]:
        """
        Convert raw article dicts into LangChain Document objects.

        topic_key: machine-readable key ("machine_learning_ai")
                   Used in metadata for filtering
        topic_display: human-readable name ("Machine Learning & AI")
                       Used in the content for LLM context

        CONTENT FORMAT:
        We combine title + abstract into page_content.
        Including the title helps because:
        1. Title is often more searchable than abstract
        2. LLM sees the title when reading retrieved chunks
        3. Better semantic matching for title-based queries
        """
        documents = []

        for article in articles:
            # Combine title and abstract as the searchable content
            # Format: "TITLE: ...\n\nABSTRACT: ..."
            content = (
                f"TITLE: {article['title']}\n\n"
                f"ABSTRACT: {article['abstract']}"
            )

            # Base metadata that all chunks from this article share
            base_metadata = {
                # Identity
                "pmid": article["pmid"],
                "url": article["url"],

                # Citation information
                "title": article["title"][:200],  # Chroma has metadata size limits
                "authors": article["authors"][:200],
                "journal": article["journal"][:100],

                # Filtering fields (these are what enable hybrid retrieval)
                "pub_date": article["pub_date"],
                "article_type": article["article_type"],
                "topic": topic_key,           # "machine_learning_ai" or "neurological_disorders"
                "topic_display": topic_display, # human-readable version

                # Source tracking
                "source": "pubmed",
                "ingested_at": datetime.now().strftime("%Y-%m-%d"),
            }

            # Split content into chunks
            # For short abstracts (< 500 chars), this returns just one chunk
            # For long abstracts, this returns 2-4 chunks
            chunks = self.splitter.split_text(content)

            for i, chunk_text in enumerate(chunks):
                # Each chunk gets the same metadata PLUS chunk index
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }

                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                documents.append(doc)

        logger.info(
            f"Created {len(documents)} document chunks "
            f"from {len(articles)} articles for topic '{topic_key}'"
        )
        return documents


# ================================================================
# EMBEDDING AND STORAGE
# ================================================================

def store_documents_in_chroma(
    documents: list[Document],
    collection,
    embeddings_model,
    topic_key: str,
    batch_size: int = 10
):
    """
    Generate embeddings for documents and store them in Chroma.

    WHY BATCH PROCESSING?
    Generating embeddings for 200+ documents takes time.
    We process in batches of 10 so:
    1. You can see progress
    2. If something fails, you don't lose all work
    3. Ollama has time to process each batch

    WHAT HAPPENS DURING EMBEDDING:
    For each document chunk, nomic-embed-text:
    1. Tokenizes the text (splits into word pieces)
    2. Passes through neural network layers
    3. Outputs a 768-dimensional vector
    4. This vector is the "meaning" of the text in mathematical form

    WHAT CHROMA STORES:
    For each document chunk, Chroma stores:
    - id:        unique string ID for this chunk
    - embedding: the 768-number vector
    - document:  the original text (page_content)
    - metadata:  all the metadata fields (pmid, date, topic, etc.)

    ID FORMAT: "pubmed_{pmid}_chunk_{index}"
    Why this format?
    - Unique: PMID is unique, chunk index differentiates chunks
    - Idempotent: adding same article twice uses same ID (no duplicates)
    - Readable: easy to debug which article a chunk belongs to
    """
    total = len(documents)
    stored = 0

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            f"Embedding batch {batch_num}/{total_batches} "
            f"({len(batch)} chunks)"
        )

        # Generate unique IDs for each chunk in this batch
        ids = [
            f"pubmed_{doc.metadata['pmid']}_chunk_{doc.metadata['chunk_index']}"
            for doc in batch
        ]

        # Extract text content for embedding
        texts = [doc.page_content for doc in batch]

        # Generate embeddings using nomic-embed-text via Ollama
        # embed_documents takes a list of strings and returns
        # a list of vectors (list of list of floats)
        # e.g. [0.2, -0.3, 0.8, ...] for each text
        embeddings = embeddings_model.embed_documents(texts)

        # Extract metadata for each document
        metadatas = [doc.metadata for doc in batch]

        # Store everything in Chroma in one batch operation
        # Chroma's add() method handles all the storage
        collection.add(
            ids=ids,
            embeddings=embeddings,  # the vectors
            documents=texts,         # original text
            metadatas=metadatas      # metadata for filtering
        )

        stored += len(batch)
        logger.info(
            f"Stored {stored}/{total} chunks "
            f"({(stored/total*100):.1f}% complete)"
        )

    logger.info(
        f"Successfully stored all {total} chunks for topic '{topic_key}'"
    )


# ================================================================
# MAIN INGESTION FUNCTION
# ================================================================

def run_ingestion(
    topics: list[str] = None,
    articles_per_topic: int = None,
    force_reingest: bool = False
):
    """
    Main function that orchestrates the entire ingestion pipeline.

    topics: list of search queries for PubMed
            defaults to settings from .env
    articles_per_topic: how many articles per topic
                        defaults to settings from .env
    force_reingest: if True, delete existing data and re-ingest
                    if False, skip topics that already have data
    """

    settings = get_settings()

    # Use provided args or fall back to .env settings
    if topics is None:
        topics = [
            t.strip()
            for t in settings.pubmed_topics.split(",")
            if t.strip()
        ]

    if articles_per_topic is None:
        articles_per_topic = settings.pubmed_articles_per_topic

    logger.info("=" * 60)
    logger.info("PubMed Knowledge Base Ingestion")
    logger.info("=" * 60)
    logger.info(f"Topics:            {topics}")
    logger.info(f"Articles per topic: {articles_per_topic}")
    logger.info(f"Force re-ingest:    {force_reingest}")
    logger.info("=" * 60)

    # -- Initialize all components ---------------------------------

    # PubMed API client
    pubmed = PubMedClient(api_key=settings.ncbi_api_key)

    # Chroma connection
    chroma_manager = ChromaStoreManager()
    collection = chroma_manager.get_or_create_collection()

    # Embedding model (nomic-embed-text via Ollama)
    logger.info("Loading embedding model...")
    embeddings_model = get_embeddings()
    logger.info("Embedding model ready")

    # Document processor
    processor = DocumentProcessor()

    # -- Process each topic ----------------------------------------

    total_articles = 0
    total_chunks = 0

    for topic in topics:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing topic: '{topic}'")
        logger.info(f"{'='*40}")

        # Create a clean key for metadata storage
        # "machine learning artificial intelligence" -> "machine_learning_artificial_intelligence"
        topic_key = topic.lower().replace(" ", "_").replace(",", "")

        # Check if already ingested (skip unless force_reingest)
        if not force_reingest and chroma_manager.collection_has_topic(
            collection, topic_key
        ):
            existing = collection.get(
                where={"topic": {"$eq": topic_key}},
                limit=1
            )
            logger.info(
                f"Topic '{topic_key}' already ingested. "
                f"Use --force to re-ingest. Skipping."
            )
            continue

        # If force re-ingest, delete existing data first
        if force_reingest:
            chroma_manager.delete_topic(collection, topic_key)

        # Step 1: Search PubMed for article IDs
        pmids = pubmed.search_articles(topic, max_results=articles_per_topic)

        if not pmids:
            logger.warning(f"No articles found for topic '{topic}'. Skipping.")
            continue

        # Step 2: Fetch full article details
        logger.info(f"Fetching details for {len(pmids)} articles...")
        articles = pubmed.fetch_article_details(pmids)
        logger.info(f"Successfully fetched {len(articles)} articles with abstracts")

        if not articles:
            logger.warning(f"No articles with abstracts found for '{topic}'")
            continue

        # Step 3: Create LangChain Documents with metadata
        # Create a nice display name from the search query
        topic_display = topic.title()  # "Machine Learning Artificial Intelligence"
        documents = processor.create_documents(
            articles, topic_key, topic_display
        )

        # Step 4: Store in Chroma with embeddings
        store_documents_in_chroma(
            documents=documents,
            collection=collection,
            embeddings_model=embeddings_model,
            topic_key=topic_key,
        )

        total_articles += len(articles)
        total_chunks += len(documents)

    # -- Final report ----------------------------------------------
    final_count = collection.count()

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    logger.info(f"Articles ingested this run: {total_articles}")
    logger.info(f"Chunks created this run:    {total_chunks}")
    logger.info(f"Total chunks in Chroma:     {final_count}")
    logger.info("=" * 60)

    # Save ingestion summary to a JSON file for reference
    summary = {
        "ingestion_date": datetime.now().isoformat(),
        "topics": topics,
        "articles_per_topic": articles_per_topic,
        "total_articles_this_run": total_articles,
        "total_chunks_this_run": total_chunks,
        "total_chunks_in_db": final_count,
    }

    summary_path = Path(__file__).parent.parent / "ingestion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")
    return summary


# ================================================================
# COMMAND LINE INTERFACE
# ================================================================

if __name__ == "__main__":
    """
    This block runs when you execute the script directly.
    It provides command-line argument parsing so you can
    customize ingestion without editing the code.

    USAGE:
        # Default (uses .env settings)
        python scripts/ingest_pubmed.py

        # Custom topics
        python scripts/ingest_pubmed.py --topics "diabetes,cancer"

        # Fewer articles (faster, for testing)
        python scripts/ingest_pubmed.py --limit 10

        # Force re-ingest (overwrite existing data)
        python scripts/ingest_pubmed.py --force
    """

    parser = argparse.ArgumentParser(
        description="Ingest PubMed articles into Chroma vector database"
    )

    parser.add_argument(
        "--topics",
        type=str,
        help="Comma-separated topics to ingest (overrides .env setting)",
        default=None
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Articles per topic (overrides .env setting)",
        default=None
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if topic already exists in Chroma"
    )

    args = parser.parse_args()

    # Parse topics from command line if provided
    topics = None
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    run_ingestion(
        topics=topics,
        articles_per_topic=args.limit,
        force_reingest=args.force
    )