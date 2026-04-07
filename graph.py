"""
graph.py - LangGraph Workflow Module
Defines the AI chatbot workflow using LangGraph with conditional routing.

Nodes:
    - router_node: Decides whether to use RAG, SERP, or BOTH
    - rag_node: Retrieves context from Pinecone
    - serp_node: Fetches results from Google via SerpAPI
    - llm_node: Generates the final answer using Groq

Routing Logic:
    - Debales AI related queries → RAG
    - General knowledge queries → SERP
    - Comparison queries (vs, compare, difference) → BOTH
"""

from typing import TypedDict, Literal
import concurrent.futures
from langgraph.graph import StateGraph, END
from rag import retrieve_from_pinecone
from serp import search_google
from llm import generate_answer


# --- State Definition ---
class ChatState(TypedDict):
    """
    The shared state passed between all nodes in the graph.
    
    Fields:
        query: The user's original question.
        route: The routing decision (RAG / SERP / BOTH).
        rag_context: Context retrieved from Pinecone.
        serp_context: Context retrieved from Google Search.
        combined_context: Merged context sent to the LLM.
        answer: The final generated answer.
        confidence_score: Similarity score from RAG (0.0 - 1.0).
        rag_documents: Individual RAG results with scores.
        serp_results: Individual SERP results with links.
    """
    query: str
    route: str
    rag_context: str
    serp_context: str
    combined_context: str
    answer: str
    confidence_score: float
    rag_documents: list
    serp_results: list


# --- Keywords: Debales-specific topics → RAG ---
DEBALES_KEYWORDS = [
    "debales", "debales ai", "debales.ai",
    "logistics automation", "ai agent", "ai agents",
    "email ai agent", "support ai agent", "sms ai agent", "phone ai agent",
    "freight broker", "3pl", "carrier", "supply chain",
    "messometer", "mess-o-meter",
    "load building", "quoting", "auto-resolve",
    "tms", "wms", "erp",
]

# --- Keywords: Comparison intent → BOTH ---
COMPARISON_KEYWORDS = [
    "compare", "comparison", "vs", "versus",
    "difference", "differences", "better",
    "which is better", "compared to", "alternative",
]

# --- Keywords: External tools/brands that need SERP context → BOTH when mixed with Debales ---
EXTERNAL_ENTITY_KEYWORDS = [
    "chatgpt", "openai", "google", "claude", "gemini",
    "hubspot", "salesforce", "microsoft", "ai tools",
    "gpt", "llm", "other tools", "competitors",
]


# ============================================================
# NODE 1: Router Node
# ============================================================
def router_node(state: ChatState) -> ChatState:
    """
    Analyze the query and decide the routing path.

    Routing rules (in priority order):
        1. Comparison keyword → BOTH
        2. External entity (ChatGPT, OpenAI, etc.) → BOTH
        3. Debales keyword + high RAG score (>= 0.5) → RAG
        4. Debales keyword + low RAG score → BOTH
        5. No Debales match → SERP
    """
    query_lower = state["query"].lower()

    comparison_words = ["compare", "difference", "vs", "better", "which is better"]
    external_keywords = ["chatgpt", "openai", "google", "ai tools", "groq", "llama", "ai model"]
    
    is_comparison = any(word in query_lower for word in comparison_words)
    is_external = any(word in query_lower for word in external_keywords)
    is_debales = any(kw in query_lower for kw in DEBALES_KEYWORDS)

    # Process RAG first if Debales related to get context
    if is_debales:
        # Comparison or mixed → BOTH
        if is_comparison or is_external:
            state["route"] = "BOTH"
        # Pure Debales → RAG
        else:
            state["route"] = "RAG"
        return state

    # General query → SERP
    state["route"] = "SERP"
    return state


# ============================================================
# NODE 2: RAG Node
# ============================================================
def rag_node(state: ChatState) -> ChatState:
    """
    Retrieve relevant documents from Pinecone vector database.
    
    Skips retrieval if context was already fetched in the router node
    (optimization to avoid duplicate API calls).
    """
    # Skip if already fetched during routing
    if state.get("rag_context"):
        return state
    
    # Retrieve from Pinecone
    result = retrieve_from_pinecone(state["query"], top_k=5)
    
    state["rag_context"] = result["context"]
    state["confidence_score"] = result["score"]
    state["rag_documents"] = result["documents"]
    
    return state


# ============================================================
# NODE 3: SERP Node
# ============================================================
def serp_node(state: ChatState) -> ChatState:
    """
    Fetch search results from Google via SerpAPI.
    Extracts titles and snippets from organic results.
    """
    result = search_google(state["query"], num_results=5)
    
    state["serp_context"] = result["context"]
    state["serp_results"] = result["results"]
    
    return state


# ============================================================
# NODE 4: BOTH Node (Parallel RAG + SERP)
# ============================================================
def both_node(state: ChatState) -> ChatState:
    """
    Executes RAG and SERP concurrently to reduce response time.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rag_future = executor.submit(retrieve_from_pinecone, state["query"], 5)
        serp_future = executor.submit(search_google, state["query"], 5)
        
        rag_result = rag_future.result()
        serp_result = serp_future.result()
        
    state["rag_context"] = rag_result["context"]
    state["confidence_score"] = rag_result["score"]
    state["rag_documents"] = rag_result["documents"]
    
    state["serp_context"] = serp_result["context"]
    state["serp_results"] = serp_result["results"]
    
    return state


# ============================================================
# NODE 4: LLM Node
# ============================================================
def llm_node(state: ChatState) -> ChatState:
    """
    Generate the final answer using Groq LLM.
    
    Combines context from RAG and/or SERP based on the route,
    then sends it to the LLM for grounded answer generation.
    """
    route = state["route"]
    
    # Build combined context based on route
    if route == "RAG":
        combined = state.get("rag_context", "No context available.")
    elif route == "SERP":
        serp_context = state.get("serp_context", "")
        if serp_context == "SERP_ERROR":
            state["answer"] = "External search failed. Please try again."
            return state
        combined = serp_context or "No search results available."
    elif route == "BOTH":
        rag_part = state.get("rag_context", "No internal context available.")
        serp_part = state.get("serp_context", "")
        
        if serp_part == "SERP_ERROR":
            # Fallback to RAG
            route = "RAG"
            combined = rag_part
        else:
            combined = (
                "=== Debales AI Internal Knowledge ===\n"
                f"{rag_part}\n\n"
                "=== External Search Results ===\n"
                f"{serp_part}"
            )
    else:
        combined = "No context available."
    
    state["combined_context"] = combined
    
    # Generate the answer
    state["answer"] = generate_answer(
        query=state["query"],
        context=combined,
        route=route
    )
    
    return state


# ============================================================
# ROUTING FUNCTION (Conditional Edge)
# ============================================================
def route_decision(state: ChatState) -> Literal["rag_node", "serp_node", "both_node"]:
    """
    Conditional edge function that directs the flow based on the route.
    
    Returns:
        The name of the next node to execute.
    """
    route = state["route"]
    if route == "RAG":
        return "rag_node"
    elif route == "SERP":
        return "serp_node"
    elif route == "BOTH":
        return "both_node"
    else:
        return "serp_node"


# ============================================================
# GRAPH BUILDER
# ============================================================
def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    
    Graph structure:
        START → router_node → (conditional) → rag_node → llm_node → END
                                            → serp_node → llm_node → END
                                            → both_rag → both_serp → llm_node → END
    """
    # Create the state graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("router_node", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("serp_node", serp_node)
    graph.add_node("both_node", both_node)
    graph.add_node("llm_node", llm_node)
    
    # Set entry point
    graph.set_entry_point("router_node")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "rag_node": "rag_node",
            "serp_node": "serp_node",
            "both_node": "both_node",
        }
    )
    
    # Add edges: after retrieval, go to LLM
    graph.add_edge("rag_node", "llm_node")
    graph.add_edge("serp_node", "llm_node")
    graph.add_edge("both_node", "llm_node")
    
    # After LLM, END
    graph.add_edge("llm_node", END)
    
    # Compile the graph
    return graph.compile()


def run_chatbot(query: str) -> dict:
    """
    Run the full chatbot pipeline for a given query.
    
    Args:
        query: The user's question.
    
    Returns:
        The final state dictionary with all results.
    """
    # Build the compiled graph
    app = build_graph()
    
    # Initialize state with defaults
    initial_state: ChatState = {
        "query": query,
        "route": "",
        "rag_context": "",
        "serp_context": "",
        "combined_context": "",
        "answer": "",
        "confidence_score": 0.0,
        "rag_documents": [],
        "serp_results": [],
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    return result
