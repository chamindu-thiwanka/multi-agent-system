"""
test_tools.py - Quick test for search and reddit tools
Run from backend/ folder: python test_tools.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_search():
    print("\n" + "="*50)
    print("Testing Internet Search Tool")
    print("="*50)
    from tools.search_tool import search_internet
    result = search_internet.invoke({
        "query": "machine learning neurological disorder diagnosis 2024",
        "max_results": 3
    })
    print(result[:1000])
    print("\nSearch tool: OK")

def test_reddit():
    print("\n" + "="*50)
    print("Testing Reddit Tool")
    print("="*50)
    from tools.reddit_tool import search_reddit
    result = search_reddit.invoke({
        "query": "machine learning medical diagnosis",
        "max_posts": 3,
        "include_comments": False
    })
    print(result[:1000])
    print("\nReddit tool: OK")

if __name__ == "__main__":
    test_search()
    test_reddit()