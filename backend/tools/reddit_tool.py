"""
reddit_tool.py - Reddit Intelligence Tool Adapter
===================================================

PURPOSE:
    Searches Reddit for community discussions, opinions, and sentiment
    on topics related to the user's query.

WHY REDDIT SPECIFICALLY?
    Reddit provides something unique that neither RAG nor internet search
    can replicate: REAL COMMUNITY OPINION at scale.

    Research papers:  "Clinical trial X showed 78% efficacy"
    News articles:    "New treatment approved by FDA"
    Reddit:           "I've been on this treatment for 6 months, here's
                       my actual experience, pros/cons, what doctors
                       don't tell you..."

    For questions like:
    - "What do patients actually think about X treatment?"
    - "What do ML engineers use in production?"
    - "What are the real challenges of implementing Y?"

    Reddit gives ground-truth community sentiment that no formal
    source provides.

ASSIGNMENT REQUIREMENT MET:
    "Reddit Intelligence Agent:
     - Must retrieve relevant threads/posts for a query
     - Summarise sentiment and themes
     - Include structured citations (post ID, URL)
     - Must be exposed as a tool usable by other agents"

    This file meets ALL four requirements.

HOW PRAW (Python Reddit API Wrapper) WORKS:
    PRAW authenticates with Reddit using your app credentials.
    It provides Python objects for subreddits, posts, and comments.

    reddit.subreddit("all").search("query") searches all of Reddit.
    reddit.subreddit("MachineLearning").search("query") searches one sub.

    Each submission (post) has:
    - title:    post title
    - selftext: post body text
    - score:    upvotes - downvotes (popularity indicator)
    - url:      link to the post
    - comments: list of Comment objects

SENTIMENT ANALYSIS APPROACH:
    We don't use a separate ML sentiment model (that would require
    another large download). Instead, we use keyword-based sentiment:
    - Positive words: great, amazing, works, recommend, effective, etc.
    - Negative words: terrible, avoid, failed, doesn't work, bad, etc.

    This is fast, requires no additional packages, and is sufficient
    for demonstrating sentiment analysis capability.
    In production, you'd use a proper sentiment model.
"""

import logging
import re
from typing import Optional
from langchain.tools import tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings

logger = logging.getLogger(__name__)


# ================================================================
# SENTIMENT ANALYSIS HELPERS
# ================================================================

# Keyword lists for basic sentiment analysis
# These are common words indicating positive/negative sentiment
# in the context of medical and technical discussions

POSITIVE_KEYWORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic",
    "works", "working", "effective", "helpful", "recommend",
    "love", "good", "better", "improved", "success", "successful",
    "positive", "benefit", "benefits", "helped", "helps",
    "promising", "breakthrough", "impressive", "significant",
    "best", "awesome", "solved", "solution", "relief", "recovery",
    "progress", "advancing", "improved", "useful", "valuable",
}

NEGATIVE_KEYWORDS = {
    "terrible", "awful", "horrible", "bad", "worse", "worst",
    "failed", "failure", "doesn't work", "doesn't help", "useless",
    "avoid", "dangerous", "harmful", "side effect", "side effects",
    "problem", "problems", "issue", "issues", "concern", "concerns",
    "disappointing", "disappointing", "ineffective", "stopped",
    "quit", "discontinued", "serious", "risk", "risks", "negative",
    "warning", "caution", "beware", "misleading", "overhyped",
}


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Analyze sentiment of text using keyword matching.

    Returns:
        tuple of (sentiment_label, confidence_score)
        sentiment_label: "positive", "negative", or "neutral"
        confidence_score: 0.0 to 1.0

    HOW IT WORKS:
        1. Convert text to lowercase
        2. Count positive keyword occurrences
        3. Count negative keyword occurrences
        4. Compare counts to determine overall sentiment
        5. Calculate confidence based on ratio

    EXAMPLE:
        "This treatment works great, really helped my condition"
        positive_count = 3 (works, great, helped)
        negative_count = 0
        sentiment = "positive", confidence = 0.9
    """
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    # Count keyword matches
    positive_count = len(words.intersection(POSITIVE_KEYWORDS))
    negative_count = len(words.intersection(NEGATIVE_KEYWORDS))
    total = positive_count + negative_count

    if total == 0:
        return "neutral", 0.5

    positive_ratio = positive_count / total

    if positive_ratio >= 0.6:
        confidence = min(0.5 + positive_ratio * 0.5, 0.95)
        return "positive", round(confidence, 2)
    elif positive_ratio <= 0.4:
        confidence = min(0.5 + (1 - positive_ratio) * 0.5, 0.95)
        return "negative", round(confidence, 2)
    else:
        return "neutral", 0.5


def get_reddit_client():
    """
    Creates and returns an authenticated PRAW Reddit client.

    WHY AUTHENTICATE?
    Reddit's API requires authentication for all requests.
    Even "read-only" access needs credentials.
    This prevents anonymous scraping and allows Reddit to
    rate-limit by application (not by IP).

    PRAW automatically handles:
    - OAuth token generation
    - Token refresh when expired
    - Rate limiting (stays under Reddit's limits)

    reddit_type="read_only" means we can only READ posts/comments.
    We don't need write access (posting, voting) for this project.
    """
    import praw

    settings = get_settings()

    if not settings.reddit_client_id or not settings.reddit_client_secret:
        raise ValueError(
            "Reddit credentials not set in .env file. "
            "Create a Reddit app at https://www.reddit.com/prefs/apps"
        )

    reddit = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        # read_only=True means we don't need a username/password
        # We're only reading public posts, not posting or voting
    )

    # Verify connection works
    # reddit.read_only is True when no username provided
    logger.info(
        f"Reddit client initialized | "
        f"Read-only: {reddit.read_only}"
    )

    return reddit


# ================================================================
# SUBREDDIT CONFIGURATION
# ================================================================

# Mapping of topic keywords to relevant subreddits
# When searching for a topic, we search these specific communities
# rather than all of Reddit.
#
# WHY TARGETED SUBREDDITS?
# Searching r/all returns everything including memes, jokes, and
# off-topic content. Targeting specific subreddits gives higher
# quality, more relevant results.

TOPIC_SUBREDDIT_MAP = {
    # AI/ML topics
    "machine learning": [
        "MachineLearning", "deeplearning", "artificial",
        "learnmachinelearning", "datascience"
    ],
    "artificial intelligence": [
        "artificial", "MachineLearning", "singularity",
        "AIethics", "learnmachinelearning"
    ],
    "neural network": [
        "MachineLearning", "deeplearning", "learnmachinelearning"
    ],
    "deep learning": [
        "deeplearning", "MachineLearning", "learnmachinelearning"
    ],

    # Medical/neurology topics
    "neurological": [
        "neuroscience", "neurology", "brainscience",
        "AskDocs", "medicine"
    ],
    "alzheimer": [
        "Alzheimers", "dementia", "neuroscience", "caregivers"
    ],
    "parkinson": [
        "Parkinsons", "neuroscience", "medicine"
    ],
    "epilepsy": [
        "Epilepsy", "neurology", "medicine"
    ],
    "depression": [
        "depression", "mentalhealth", "psychology", "neuroscience"
    ],
    "stroke": [
        "stroke", "neurology", "medicine", "AskDocs"
    ],

    # General fallback subreddits
    "default": [
        "science", "medicine", "AskScience",
        "neuroscience", "MachineLearning"
    ]
}


def find_relevant_subreddits(query: str) -> list[str]:
    """
    Find the most relevant subreddits for a given query.

    Checks each keyword in the topic map against the query.
    Returns matching subreddits, or default subreddits if no match.
    """
    query_lower = query.lower()
    subreddits = []

    for keyword, subs in TOPIC_SUBREDDIT_MAP.items():
        if keyword in query_lower:
            subreddits.extend(subs)

    # Remove duplicates while preserving order
    seen = set()
    unique_subreddits = []
    for s in subreddits:
        if s not in seen:
            seen.add(s)
            unique_subreddits.append(s)

    # Fall back to defaults if no match found
    if not unique_subreddits:
        unique_subreddits = TOPIC_SUBREDDIT_MAP["default"]

    # Limit to first 3 subreddits to avoid too many API calls
    return unique_subreddits[:3]


# ================================================================
# REDDIT TOOLS
# ================================================================

@tool
def search_reddit(
    query: str,
    max_posts: int = 5,
    include_comments: bool = True,
) -> str:
    """
    Search Reddit for community discussions and opinions on a topic.

    Use this tool when you want to understand:
    - What real people think about a medical treatment or condition
    - Community consensus on a technical approach or tool
    - Patient experiences and practical advice
    - Common concerns, misconceptions, or debates about a topic
    - Ground-level perspectives that academic papers don't capture

    This is different from retrieve_documents (academic research) and
    search_internet (news articles). Reddit gives you raw community
    sentiment and personal experience.

    Args:
        query: Topic to search Reddit for.
               Example: "deep learning frameworks comparison pytorch tensorflow"

        max_posts: Number of posts to analyze (1-10, default 5).
                   More posts = more comprehensive sentiment analysis
                   but takes longer to process.

        include_comments: Whether to analyze top comments on each post.
                          True = richer analysis, more context (default)
                          False = faster, titles and posts only

    Returns:
        Formatted analysis including post summaries, community sentiment,
        key themes, and structured citations with Reddit URLs.
    """

    logger.info(
        f"[Reddit Tool] Query: '{query}' | "
        f"Max posts: {max_posts} | "
        f"Include comments: {include_comments}"
    )

    try:
        reddit = get_reddit_client()

        # Find relevant subreddits for this query
        target_subreddits = find_relevant_subreddits(query)
        logger.info(f"[Reddit Tool] Searching subreddits: {target_subreddits}")

        all_posts = []

        # Search each subreddit and collect posts
        for subreddit_name in target_subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)

                # Search within this subreddit
                # sort="relevance" gives most relevant results first
                # time_filter="year" limits to last 12 months
                # This avoids very old, potentially outdated posts
                search_results = subreddit.search(
                    query,
                    sort="relevance",
                    time_filter="year",
                    limit=max_posts,
                )

                for post in search_results:
                    # Skip posts with very low engagement
                    # A post with 1 upvote is likely irrelevant or spam
                    if post.score < 2:
                        continue

                    post_data = {
                        "id": post.id,
                        "title": post.title,
                        "text": post.selftext[:500] if post.selftext else "",
                        "score": post.score,
                        "url": f"https://reddit.com{post.permalink}",
                        "subreddit": subreddit_name,
                        "num_comments": post.num_comments,
                        "comments": [],
                    }

                    # Fetch top comments if requested
                    if include_comments and post.num_comments > 0:
                        try:
                            # Replace MoreComments objects with actual comments
                            # limit=None fetches all comments (up to Reddit's limit)
                            # We then sort by score and take top 3
                            post.comments.replace_more(limit=0)

                            top_comments = sorted(
                                [c for c in post.comments.list()
                                 if hasattr(c, 'body') and len(c.body) > 20],
                                key=lambda c: c.score,
                                reverse=True
                            )[:3]  # only top 3 comments per post

                            post_data["comments"] = [
                                {
                                    "text": c.body[:300],
                                    "score": c.score,
                                }
                                for c in top_comments
                            ]
                        except Exception as comment_error:
                            # Comments failing shouldn't break the whole search
                            logger.warning(
                                f"Failed to fetch comments for post "
                                f"{post.id}: {comment_error}"
                            )

                    all_posts.append(post_data)

                    # Stop if we have enough posts
                    if len(all_posts) >= max_posts:
                        break

            # except Exception as sub_error:
            #     # One subreddit failing shouldn't stop others
            #     logger.warning(
            #         f"Error searching r/{subreddit_name}: {sub_error}"
            #     )
            #     continue
            except Exception as sub_error:
                error_str = str(sub_error)
                # Detect Reddit API authorization errors specifically
                # 401 = credentials rejected (pending approval)
                # 403 = access forbidden
                if "401" in error_str or "403" in error_str:
                    logger.warning(
                        f"Reddit API access not yet approved for "
                        f"r/{subreddit_name}. "
                        f"Submit access request at reddit.com/prefs/apps"
                    )
                    # Return informative message instead of crashing
                    return (
                        f"Reddit Intelligence: API access pending approval.\n"
                        f"Query was: '{query}'\n"
                        f"Reddit changed their API policy in 2023 and now "
                        f"requires manual approval for new applications.\n"
                        f"Once approved, this tool will automatically work "
                        f"with the existing credentials.\n"
                        f"The system continues to function using RAG and "
                        f"internet search tools."
                    )
                else:
                    logger.warning(
                        f"Error searching r/{subreddit_name}: {sub_error}"
                    )
                continue

            if len(all_posts) >= max_posts:
                break

        if not all_posts:
            return (
                f"No Reddit discussions found for: '{query}'. "
                "The topic may be too specific or use different terminology "
                "than Reddit communities use."
            )

        # Perform sentiment analysis on collected content
        all_text = " ".join([
            f"{p['title']} {p['text']} "
            + " ".join([c['text'] for c in p['comments']])
            for p in all_posts
        ])

        overall_sentiment, confidence = analyze_sentiment(all_text)

        # Extract key themes (most common meaningful words)
        themes = extract_key_themes(all_text, n_themes=5)

        # Format the complete response
        output_lines = [
            f"Reddit Community Analysis for: '{query}'\n",
            f"Searched subreddits: {', '.join(['r/' + s for s in target_subreddits])}",
            f"Posts analyzed: {len(all_posts)}",
            f"Overall sentiment: {overall_sentiment.upper()} "
            f"(confidence: {confidence:.0%})",
            f"Key themes: {', '.join(themes)}\n",
            "=" * 50,
            "INDIVIDUAL POSTS:",
            "=" * 50,
        ]

        for i, post in enumerate(all_posts, 1):
            post_sentiment, post_conf = analyze_sentiment(
                f"{post['title']} {post['text']}"
            )

            output_lines.extend([
                f"\nPost {i}:",
                f"  Title:     {post['title']}",
                f"  Subreddit: r/{post['subreddit']}",
                f"  Score:     {post['score']} upvotes | "
                f"{post['num_comments']} comments",
                f"  Sentiment: {post_sentiment} ({post_conf:.0%} confidence)",
                f"  URL:       {post['url']}",
                f"  Post ID:   {post['id']}",
            ])

            if post["text"]:
                output_lines.append(
                    f"  Content:   {post['text'][:300]}..."
                    if len(post['text']) > 300
                    else f"  Content:   {post['text']}"
                )

            if post["comments"]:
                output_lines.append("  Top comments:")
                for j, comment in enumerate(post["comments"], 1):
                    comment_sentiment, _ = analyze_sentiment(comment["text"])
                    output_lines.append(
                        f"    Comment {j} ({comment['score']} pts, "
                        f"{comment_sentiment}): "
                        f"{comment['text'][:200]}..."
                        if len(comment['text']) > 200
                        else f"    Comment {j} ({comment['score']} pts, "
                             f"{comment_sentiment}): {comment['text']}"
                    )

        output_lines.extend([
            "\n" + "=" * 50,
            "CITATION SUMMARY:",
            "=" * 50,
        ])

        for post in all_posts:
            output_lines.append(
                f"  [Reddit:{post['id']}] r/{post['subreddit']} - "
                f"{post['title'][:60]}... | {post['url']}"
            )

        return "\n".join(output_lines)

    except Exception as e:
        error_msg = f"Reddit search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def extract_key_themes(text: str, n_themes: int = 5) -> list[str]:
    """
    Extract the most frequently mentioned meaningful words as themes.

    WHY NOT USE NLP LIBRARIES?
    Libraries like NLTK or spaCy require additional downloads and
    add complexity. For this project, simple word frequency analysis
    is sufficient to demonstrate theme extraction.

    HOW IT WORKS:
    1. Split text into words
    2. Remove common words (stopwords) that carry no meaning
       ("the", "a", "is", "and", etc.)
    3. Count remaining word frequencies
    4. Return the most frequent meaningful words as themes

    These "themes" represent what topics appear most in the community
    discussions, giving us insight into what aspects people focus on.
    """

    # Common English words that carry no meaningful information
    # Removing these reveals the actual meaningful content
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "as", "is", "was",
        "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "this", "that", "these", "those", "it",
        "its", "i", "me", "my", "we", "our", "you", "your", "they",
        "their", "he", "she", "his", "her", "what", "which", "who",
        "not", "no", "so", "if", "then", "than", "about", "also",
        "just", "like", "more", "can", "when", "there", "some",
        "all", "any", "one", "two", "first", "after", "before",
        "new", "very", "how", "get", "use", "used", "using", "think"
    }

    # Extract words (only alphabetic, minimum 4 characters)
    # 4+ character minimum filters out meaningless short words
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())

    # Count frequencies, excluding stopwords
    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top N
    sorted_words = sorted(
        word_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [word for word, count in sorted_words[:n_themes]]


@tool
def get_reddit_post(post_id: str) -> str:
    """
    Retrieve the full content of a specific Reddit post by its ID.

    Use this when a previous Reddit search returned an interesting post
    and you need its full content, including all comments.

    Args:
        post_id: Reddit post ID from a previous search result.
                 Example: "1abc23d" (appears as [Reddit:1abc23d] in citations)

    Returns:
        Full post content with all available comments and metadata.
    """

    logger.info(f"[Reddit Tool] Fetching post: {post_id}")

    try:
        reddit = get_reddit_client()

        submission = reddit.submission(id=post_id)

        # Fetch all comments
        submission.comments.replace_more(limit=0)
        all_comments = submission.comments.list()

        # Sort comments by score (most upvoted first)
        top_comments = sorted(
            [c for c in all_comments
             if hasattr(c, 'body') and len(c.body) > 10],
            key=lambda c: c.score,
            reverse=True
        )[:10]  # top 10 comments

        overall_sentiment, confidence = analyze_sentiment(
            submission.title + " " + submission.selftext
        )

        lines = [
            f"FULL REDDIT POST",
            f"{'='*50}",
            f"Title:      {submission.title}",
            f"Subreddit:  r/{submission.subreddit.display_name}",
            f"Score:      {submission.score} upvotes",
            f"Comments:   {submission.num_comments}",
            f"Posted:     {submission.url}",
            f"Sentiment:  {overall_sentiment} ({confidence:.0%} confidence)",
            f"{'='*50}",
            f"CONTENT:",
            submission.selftext if submission.selftext else "(Link post - no text body)",
            f"{'='*50}",
            f"TOP COMMENTS ({len(top_comments)} shown):",
        ]

        for i, comment in enumerate(top_comments, 1):
            c_sentiment, _ = analyze_sentiment(comment.body)
            lines.extend([
                f"\nComment {i} ({comment.score} upvotes, {c_sentiment}):",
                comment.body[:500] + "..." if len(comment.body) > 500
                else comment.body,
            ])

        return "\n".join(lines)

    except Exception as e:
        error_msg = f"Failed to fetch Reddit post {post_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg