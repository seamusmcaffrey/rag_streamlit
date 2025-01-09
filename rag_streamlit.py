import os
import voyageai
from pinecone import Pinecone
from anthropic import Client
import streamlit as st
import json

# Set your API keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initializations
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "rag-index"
index = pc.Index(INDEX_NAME)
claude = Client(api_key=CLAUDE_API_KEY)

def init_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

def retrieve_context(prompt):
    """Query Pinecone for relevant context"""
    try:
        st.write("Debug: Starting context retrieval")
        
        # Get embeddings
        st.write("Debug: Generating embeddings")
        query_embedding = voyage.embed([prompt], model="voyage-3", input_type="query").embeddings[0]
        st.write("Debug: Embeddings generated successfully")
        
        # Query Pinecone
        st.write("Debug: Querying Pinecone")
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        st.write(f"Debug: Received {len(results['matches'])} matches from Pinecone")
        
        # Extract contexts
        contexts = [item["metadata"].get("content", "") for item in results["matches"]]
        st.write(f"Debug: Extracted {len(contexts)} context pieces")
        
        final_context = "\n\n".join(contexts)
        st.write(f"Debug: Final context length: {len(final_context)} characters")
        
        return final_context
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        st.write(f"Debug: Full error details: {repr(e)}")
        return ""

def fetch_claude_response(prompt, context, max_tokens=1000):
    """Fetch response from Claude API"""
    try:
        # Log the incoming request
        st.write("Debug: Preparing to send request to Claude")
        
        structured_prompt = f"""Context:
{context}

Question:
{prompt}

Please provide a clear and concise response focusing on boardgame.io implementation details."""

        # Log the structured prompt
        st.write(f"Debug: Context length: {len(context)} characters")
        
        response = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            system="You are an expert coder specialized in the boardgame.io library. Provide specific, actionable advice and code examples when appropriate.",
            messages=[
                {"role": "user", "content": structured_prompt}
            ]
        )
        
        # Log the raw response for debugging
        st.write("Debug: Received response from Claude")
        st.write(f"Debug: Response type: {type(response)}")
        st.write(f"Debug: Response attributes: {dir(response)}")
        
        # Extract the content from the response
        if hasattr(response, 'content'):
            st.write(f"Debug: Content type: {type(response.content)}")
            # Handle the case where content might be a list of content blocks
            if isinstance(response.content, list):
                content = "\n".join(str(block) for block in response.content)
                st.write(f"Debug: Joined content from list: {len(content)} characters")
                return content
            st.write(f"Debug: Single content block: {len(str(response.content))} characters")
            return str(response.content)
        
        st.write("Debug: No content attribute found in response")
        return "I apologize, but I couldn't generate a response at this time."
    
    except Exception as e:
        st.error(f"Error getting Claude's response: {str(e)}")
        return "I encountered an error while processing your request."

def format_message(message):
    """Format message for display, handling code blocks"""
    if "```" in message:
        parts = message.split("```")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                if part.strip():
                    formatted_parts.append(part)
            else:  # Code block
                language = "python" if part.startswith("python\n") else ""
                code = part.replace("python\n", "") if language else part
                st.code(code.strip(), language=language)
        return " ".join(formatted_parts)
    return message

def main():
    st.set_page_config(page_title="RAG-Assisted Claude Chat", layout="wide")
    st.title("RAG-Assisted Claude Chat")
    st.markdown("A conversational AI powered by RAG-assisted Claude, specialized in boardgame.io")
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    
    # Configure debug logging
    if debug_mode:
        st.write("Debug Mode Enabled")
        st.write(f"Session State Keys: {list(st.session_state.keys())}")
        st.write(f"Environment Variables Set: VOYAGE_API_KEY={'VOYAGE_API_KEY' in os.environ}, "
                f"CLAUDE_API_KEY={'CLAUDE_API_KEY' in os.environ}, "
                f"PINECONE_API_KEY={'PINECONE_API_KEY' in os.environ}, "
                f"PINECONE_ENV={'PINECONE_ENV' in os.environ}")
    else:
        # If debug mode is off, override the debug write statements
        def debug_write(*args, **kwargs):
            pass
        st.write = debug_write
    
    init_session_state()
    
    # Chat display
    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker.lower()):
            if speaker == "Claude":
                format_message(message)
            else:
                st.write(message)
    
    # User input
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append(("User", prompt))
        
        with st.chat_message("claude"):
            with st.spinner("Thinking..."):
                context = retrieve_context(prompt)
                response = fetch_claude_response(prompt, context)
                st.session_state.chat_history.append(("Claude", response))
                format_message(response)

if __name__ == "__main__":
    main()