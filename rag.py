"""
rag.py - Pinecone Retrieval Module
Handles vector similarity search using Sentence Transformers (all-MiniLM-L6-v2).
Includes dual filtering: score threshold + noise keyword removal + business relevance check.
"""

import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Initialize Pinecone client ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# --- Connect to the existing index (do NOT change this) ---
INDEX_NAME = "debales-ai"
index = pc.Index(INDEX_NAME)

# --- Load the embedding model (do NOT change this) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Filter: words that indicate noisy/irrelevant content ---
NOISE_KEYWORDS = [
    "python", "learning", "student", "testimonial",
    "review", "experience", "miniature", "wellness",
    "floral", "dining", "menu", "cooking", "recipe",
    "cart", "skyrocketed", "thrilled",
]

# --- Filter: words that indicate relevant business content ---
BUSINESS_KEYWORDS = [
    "logistics", "automation", "ai agent", "supply chain",
    "freight", "order", "shipment", "broker", "carrier",
    "3pl", "email", "quote", "dispatch", "tracking",
    "debales", "automate", "workflow", "tms", "wms", "erp",
    "eta", "invoice", "warehouse", "operations",
]


def embed_query(query: str) -> list:
    """
    Embed a text query into a 384-dimensional vector using MiniLM.

    Args:
        query: The user's question as a string.

    Returns:
        A list of floats representing the query embedding.
    """
    return embed_model.encode(query).tolist()


def is_noise(text: str) -> bool:
    """
    Return True if the text contains noise keywords (should be excluded).

    Args:
        text: Document text to check.

    Returns:
        True if noisy, False if clean.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in NOISE_KEYWORDS)


def is_business_relevant(text: str) -> bool:
    """
    Return True if the text contains at least one business-relevant keyword.

    Args:
        text: Document text to check.

    Returns:
        True if relevant, False otherwise.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in BUSINESS_KEYWORDS)


def retrieve_from_pinecone(query: str, top_k: int = 8) -> dict:
    """
    Retrieve relevant documents from Pinecone with dual filtering.

    Steps:
        1. Embed the query using Sentence Transformers.
        2. Query Pinecone for top_k most similar vectors.
        3. Extract text and score from metadata.
        4. Apply dual filter: score > 0.4 AND no noise AND business relevant.
        5. Fallback to top 3 raw results if nothing passes the filter.
        6. Limit to top 5 documents.

    Args:
        query: The user's question.
        top_k: Number of raw results to fetch from Pinecone (default: 8).

    Returns:
        {
            "context": str,    # Combined text from filtered docs
            "score": float,    # Best similarity score
            "documents": list  # List of {"text": str, "score": float}
        }
    """
    # Step 1: Embed query
    query_vector = embed_query(query)

    # Step 2: Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Step 3: Extract all documents with text and score
    raw_docs = []
    for match in results.get("matches", []):
        doc_text = match.get("metadata", {}).get("text", "").strip()
        doc_score = match.get("score", 0.0)
        if doc_text:
            raw_docs.append({
                "text": doc_text,
                "score": round(doc_score, 4)
            })

    # Step 4: Apply progressive filtering
    # Step 4.1: Strict filtering
    filtered_docs = [
        doc for doc in raw_docs
        if doc["score"] > 0.4
        and not is_noise(doc["text"])
        and is_business_relevant(doc["text"])
    ]

    # Step 4.2: Relax filtering if too few results
    if len(filtered_docs) < 2:
        filtered_docs = [
            doc for doc in raw_docs
            if doc["score"] > 0.4 and not is_noise(doc["text"])
        ]

    # Step 4.3: Fallback if empty
    if not filtered_docs:
        filtered_docs = raw_docs[:3]

    # Step 6: Limit to top 5 documents
    filtered_docs = filtered_docs[:5]

    # Build combined context string from filtered documents
    context = "\n\n".join([doc["text"] for doc in filtered_docs])

    # Best score is from the highest-ranked document
    best_score = filtered_docs[0]["score"] if filtered_docs else 0.0

    return {
        "context": context,
        "score": best_score,
        "documents": filtered_docs
    }
