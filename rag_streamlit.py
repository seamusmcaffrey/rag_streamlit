import os
import voyageai
from pinecone import Pinecone
from anthropic import Client
import streamlit as st
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'chatbot_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize API clients
def init_clients():
    """Initialize API clients with error handling"""
    try:
        logger.info("Initializing API clients...")
        
        # Check for required environment variables
        required_vars = ["VOYAGE_API_KEY", "CLAUDE_API_KEY", "PINECONE_API_KEY", "PINECONE_ENV"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        claude = Client(api_key=os.getenv("CLAUDE_API_KEY"))
        voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        index = pc.Index("rag-index")
        
        logger.info("API clients initialized successfully")
        return claude, voyage, index
    
    except Exception as e:
        logger.error(f"Error initializing clients: {str(e)}")
        st.error(f"Error initializing clients: {str(e)}")
        return None, None, None

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized empty messages list in session state")

def get_rag_context(voyage_client, pinecone_index, query):
    """Get relevant context from RAG system"""
    try:
        logger.debug(f"Getting RAG context for query: {query}")
        embedding = voyage_client.embed([query], model="voyage-3", input_type="query").embeddings[0]
        results = pinecone_index.query(vector=embedding, top_k=3, include_metadata=True)
        context = "\n\n".join([match["metadata"].get("content", "") for match in results["matches"]])
        logger.debug(f"Retrieved context length: {len(context)}")
        return context
    except Exception as e:
        logger.error(f"Error getting RAG context: {str(e)}")
        return ""

def get_assistant_response(claude_client, prompt, context=""):
    """Get response from Claude"""
    try:
        logger.debug(f"Getting Claude response for prompt: {prompt}")
        logger.debug(f"Context length: {len(context)}")
        
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
        
        logger.debug(f"Got response from Claude: {response.content}")
        return response.content
    
    except Exception as e:
        logger.error(f"Error getting Claude response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def main():
    st.set_page_config(page_title="RAG-Assisted Claude Chat", layout="wide")
    
    # Initialize clients
    claude, voyage, index = init_clients()
    if not all([claude, voyage, index]):
        st.error("Failed to initialize required clients. Check the logs for details.")
        return
    
    st.title("RAG-Assisted Claude Chat")
    st.markdown("A conversational AI powered by RAG-assisted Claude, specialized in boardgame.io")
    
    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode")
    if debug_mode:
        st.sidebar.text("Session State:")
        st.sidebar.write(st.session_state)
    
    # Initialize session state
    init_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about boardgame.io..."):
        # Log the received prompt
        logger.info(f"Received prompt: {prompt}")
        
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get RAG context if relevant
                context = get_rag_context(voyage, index, prompt)
                
                # Get Claude's response
                response = get_assistant_response(claude, prompt, context)
                
                # Display response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                if debug_mode:
                    st.sidebar.text("Latest Context Length:")
                    st.sidebar.write(len(context))

if __name__ == "__main__":
    main()