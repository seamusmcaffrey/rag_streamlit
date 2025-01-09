import os
import voyageai
from pinecone import Pinecone
from anthropic import Client
import streamlit as st
import logging
from datetime import datetime

# Minimize logging output
logging.getLogger().setLevel(logging.WARNING)
for log_name in ['streamlit', 'watchdog.observers.inotify_buffer']:
    logging.getLogger(log_name).setLevel(logging.WARNING)

@st.cache_resource
def init_clients():
    """Initialize API clients with error handling"""
    try:
        # Check for required environment variables
        required_vars = ["VOYAGE_API_KEY", "CLAUDE_API_KEY", "PINECONE_API_KEY", "PINECONE_ENV"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        claude = Client(api_key=os.getenv("CLAUDE_API_KEY"))
        voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        index = pc.Index("rag-index")
        
        return claude, voyage, index
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None, None

def format_response(response):
    """Format the response for display, handling code blocks properly"""
    if isinstance(response, str):
        if "```" in response:
            parts = response.split("```")
            formatted_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Regular text
                    if part.strip():
                        formatted_parts.append(part.strip())
                else:  # Code block
                    lines = part.strip().split('\n')
                    if lines and lines[0] in ['python', 'javascript', 'typescript']:
                        language = lines[0]
                        code = '\n'.join(lines[1:])
                    else:
                        language = ''
                        code = part
                    st.code(code.strip(), language=language)
            return "\n\n".join(formatted_parts)
        return response
    return str(response)

@st.cache_data(ttl="1h")
def get_rag_context(query, _voyage_client, _pinecone_index):
    """Get relevant context from RAG system"""
    try:
        embedding = _voyage_client.embed([query], model="voyage-3", input_type="query").embeddings[0]
        results = _pinecone_index.query(vector=embedding, top_k=3, include_metadata=True)
        return "\n\n".join([match["metadata"].get("content", "") for match in results["matches"]])
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

def get_assistant_response(prompt, context, claude_client, message_history):
    """Get response from Claude with conversation history"""
    try:
        # Prepare the full message history
        messages = []
        
        # Add previous messages, excluding system messages
        for msg in message_history:
            if msg["role"] != "system":
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current prompt with context
        messages.append({
            "role": "user",
            "content": f"{context}\n\nUser: {prompt}" if context else prompt
        })
        
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            system="You are a helpful AI assistant with expertise in boardgame.io. Engage naturally with users, providing technical details only when specifically asked. Remember to keep track of the conversation context and refer back to previous questions when relevant."
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
            st.markdown(format_response(message["content"]))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about boardgame.io..."):
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get RAG context
                context = get_rag_context(prompt, voyage, index)
                
                # Get response with full conversation history
                response = get_assistant_response(
                    prompt, 
                    context, 
                    claude, 
                    st.session_state.messages[-10:]  # Keep last 10 messages to manage context window
                )
                
                # Display and store response
                st.markdown(format_response(response))
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()