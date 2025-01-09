import os
import voyageai
from pinecone import Pinecone
from anthropic import Client
import streamlit as st
import logging
from datetime import datetime

# Minimal logging setup - only for errors and critical events
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s',
    force=True
)

@st.cache_resource
def init_clients():
    """Initialize API clients with error handling"""
    try:
        claude = Client(api_key=os.getenv("CLAUDE_API_KEY"))
        voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        index = pc.Index("rag-index")
        return claude, voyage, index
    except Exception as e:
        st.error(f"Failed to initialize clients: {str(e)}")
        return None, None, None

@st.cache_data(ttl="1h")
def get_rag_context(query, voyage_client, pinecone_index):
    """Get relevant context from RAG system"""
    try:
        embedding = voyage_client.embed([query], model="voyage-3", input_type="query").embeddings[0]
        results = pinecone_index.query(vector=embedding, top_k=3, include_metadata=True)
        return "\n\n".join([match["metadata"].get("content", "") for match in results["matches"]])
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

def get_assistant_response(prompt, context, claude_client):
    """Get response from Claude"""
    try:
        messages = [
            {
                "role": "user",
                "content": f"{context}\n\nUser: {prompt}" if context else prompt
            }
        ]
        
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            system="You are a helpful AI assistant with expertise in boardgame.io. Engage naturally with users, providing technical details only when specifically asked."
        )
        return response.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def main():
    st.set_page_config(page_title="RAG-Assisted Claude Chat", layout="wide")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize clients (cached)
    claude, voyage, index = init_clients()
    if not all([claude, voyage, index]):
        return
    
    st.title("RAG-Assisted Claude Chat")
    st.markdown("A conversational AI powered by RAG-assisted Claude, specialized in boardgame.io")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about boardgame.io..."):
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = get_rag_context(prompt, voyage, index)
                response = get_assistant_response(prompt, context, claude)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()