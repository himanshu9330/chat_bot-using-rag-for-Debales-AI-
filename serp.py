"""
serp.py - SerpAPI Integration Module
Uses the official serpapi Python client (google-search-results package)
to perform Google searches and extract titles + snippets from organic results.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- SerpAPI key ---
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")


def search_google(query: str, num_results: int = 5) -> dict:
    """
    Perform a Google search via SerpAPI and return structured results using requests.

    Args:
        query: The search query string.
        num_results: Number of organic results to return (default: 5).
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
    }

    try:
        # Request the API directly
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()

        # Extract organic results
        organic_results = data.get("organic_results", [])

        results = []
        for item in organic_results[:num_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")

            # Only include results that have a snippet
            if snippet:
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })

        # Build compact context string matching the notebook format
        context_parts = [f"{r['title']}: {r['snippet']}" for r in results]
        context = "\n".join(context_parts)

        if not context:
            return {
                "context": "No useful web data found.",
                "results": [],
                "source": "Google Search via SerpAPI"
            }

        return {
            "context": context,
            "results": results,
            "source": "Google Search via SerpAPI"
        }

    except Exception as e:
        # Graceful error handling
        print("SERP ERROR:", e)
        return {
            "context": "SERP_ERROR",
            "results": [],
            "source": "Google Search via SerpAPI (Error)"
        }
