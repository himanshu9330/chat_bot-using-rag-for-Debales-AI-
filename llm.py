"""
llm.py - Groq LLM Wrapper Module
Handles all interactions with the Groq API using the Llama 3.1 8B Instant model.
Uses tight, focused prompts per route to avoid hallucination and irrelevant content.
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Initialize Groq client ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Model configuration ---
MODEL_NAME = "llama-3.1-8b-instant"


def generate_answer(query: str, context: str, route: str) -> str:
    """
    Generate a grounded answer using the Groq LLM.

    Selects the appropriate prompt template based on the route:
        - RAG: Strict product-focused prompt, skips testimonials.
        - SERP: Web data only, general Q&A.
        - BOTH: Combined sources for comparison queries.

    Args:
        query: The user's original question.
        context: Retrieved context from RAG, SERP, or combined.
        route: The routing decision ("RAG", "SERP", or "BOTH").

    Returns:
        The generated answer as a string.
    """
    # Build the full prompt based on route
    prompt = _build_prompt(query, context, route)

    try:
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,   # Low temperature for factual, consistent responses
            max_tokens=1024,
            top_p=0.9,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"


def _build_prompt(query: str, context: str, route: str) -> str:
    """
    Build a focused prompt template based on the routing decision.

    RAG prompt: Instructs the model to extract core product/feature info only.
    SERP prompt: Uses web search data to answer general questions.
    BOTH prompt: Combines internal knowledge + web data for comparisons.

    Args:
        query: The user's question.
        context: The retrieved context string.
        route: "RAG", "SERP", or "BOTH".

    Returns:
        A complete prompt string ready to send to the LLM.
    """

    if route == "RAG":
        return f"""
You are an AI assistant.

Answer clearly using the context.

Focus on:
- what Debales AI does
- its main features

Do NOT say "not enough information" unless context is completely empty.

Context:
{context}

Question: {query}

Answer in 2-3 lines.
"""

    elif route == "SERP":
        return f"""You are a helpful AI assistant. Answer the following question using ONLY the web search data provided below.

Web Search Data:
{context}

Question: {query}

Provide a clear and accurate answer based on the data above.
If the data does not contain enough information, say: "I don't know based on the available information."
"""

    elif route == "BOTH":
        return f"""You are a helpful AI assistant. Answer the following question using BOTH sources provided below.

=== Debales AI Internal Knowledge ===
{context.split("=== External Search Results ===")[0] if "=== External Search Results ===" in context else context}

=== External Web Search Results ===
{context.split("=== External Search Results ===")[1] if "=== External Search Results ===" in context else "No external data available."}

Question: {query}

Instructions:
- Clearly compare or combine both sources.
- Label information from Debales AI separately from external information.
- Be concise and factual.
- If a source lacks information, say so explicitly.
- Do NOT hallucinate or make up information.
"""

    else:
        # Fallback generic prompt
        return f"""Answer the following question based on the context below.

Context:
{context}

Question: {query}

If you don't know the answer, say: "I don't know based on the available information."
"""
