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
        query_embedding = voyage.embed([prompt], model="voyage-3", input_type="query").embeddings[0]
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        contexts = [item["metadata"].get("content", "") for item in results["matches"]]
        return "\n\n".join(contexts)
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

def fetch_claude_response(prompt, context, max_tokens=1000):
    """Fetch response from Claude API"""
    try:
        structured_prompt = f"""Context:
{context}

Question:
{prompt}

Please provide a clear and concise response focusing on boardgame.io implementation details."""

        response = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            system="You are an expert coder specialized in the boardgame.io library. Provide specific, actionable advice and code examples when appropriate.",
            messages=[
                {"role": "user", "content": structured_prompt}
            ]
        )
        
        # Extract the content from the response
        if hasattr(response, 'content'):
            # Handle the case where content might be a list of content blocks
            if isinstance(response.content, list):
                return "\n".join(block.text for block in response.content if hasattr(block, 'text'))
            return response.content
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