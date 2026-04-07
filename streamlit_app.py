"""
streamlit_app.py - Streamlit Frontend for the AI Chatbot
A clean, modern chatbot UI with:
    - Chat history display
    - Route indicator (RAG / SERP / BOTH)
    - Debug sidebar with similarity scores and retrieved context
"""

import streamlit as st
from graph import run_chatbot


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Debales AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Route badges */
    .route-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .route-rag {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .route-serp {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .route-both {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 4px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }
    
    /* Header styling */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "debug_info" not in st.session_state:
    st.session_state.debug_info = None


# ============================================================
# SIDEBAR - Debug Information
# ============================================================
with st.sidebar:
    st.markdown("### 🔍 Debug Panel")
    st.markdown("---")
    
    if st.session_state.debug_info:
        info = st.session_state.debug_info
        
        # Route used
        route = info.get("route", "N/A")
        route_class = f"route-{route.lower()}"
        st.markdown(f"**Route Used:**")
        st.markdown(
            f'<span class="route-badge {route_class}">{route}</span>',
            unsafe_allow_html=True
        )
        
        # Confidence score
        score = info.get("confidence_score", 0.0)
        st.markdown(f"**Confidence Score:** `{score:.4f}`")
        st.progress(min(score, 1.0))
        
        # RAG Documents
        if info.get("rag_documents"):
            st.markdown("---")
            st.markdown("**📄 RAG Documents Retrieved:**")
            for i, doc in enumerate(info["rag_documents"], 1):
                with st.expander(f"Document {i} (Score: {doc['score']})"):
                    st.text(doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"])
        
        # SERP Results
        if info.get("serp_results"):
            st.markdown("---")
            st.markdown("**🌐 SERP Results:**")
            for i, result in enumerate(info["serp_results"], 1):
                with st.expander(f"{i}. {result['title'][:50]}..."):
                    st.markdown(f"**{result['title']}**")
                    st.text(result["snippet"])
                    if result.get("link"):
                        st.markdown(f"[🔗 Source]({result['link']})")
        
        # Combined Context
        st.markdown("---")
        with st.expander("📋 Combined Context Sent to LLM"):
            st.text(info.get("combined_context", "N/A")[:2000])
    else:
        st.info("Ask a question to see debug information here.")
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.debug_info = None
        st.rerun()


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.markdown('<p class="header-title">🤖 Debales AI Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    "Ask me anything about **Debales AI** or general logistics topics. "
    "I'll automatically route your question to the best source."
)
st.markdown("---")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Show route badge above the answer
            route = message.get("route", "")
            if route:
                route_class = f"route-{route.lower()}"
                st.markdown(
                    f'<span class="route-badge {route_class}">📍 {route}</span>',
                    unsafe_allow_html=True
                )
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask me about Debales AI or anything else...")

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Add to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
    })
    
    # Run the chatbot pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔄 Routing and generating answer..."):
            result = run_chatbot(user_query)
        
        # Show route badge
        route = result.get("route", "N/A")
        route_class = f"route-{route.lower()}"
        st.markdown(
            f'<span class="route-badge {route_class}">📍 {route}</span>',
            unsafe_allow_html=True
        )
        
        # Show the answer
        answer = result.get("answer", "Sorry, I couldn't generate a response.")
        st.markdown(answer)
    
    # Save assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "route": route,
    })
    
    # Update debug info in sidebar
    st.session_state.debug_info = {
        "route": route,
        "confidence_score": result.get("confidence_score", 0.0),
        "rag_documents": result.get("rag_documents", []),
        "serp_results": result.get("serp_results", []),
        "combined_context": result.get("combined_context", ""),
    }
    
    # Rerun to update sidebar
    st.rerun()
