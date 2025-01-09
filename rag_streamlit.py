import os
import json
import voyageai
from pinecone import Pinecone
from anthropic import Client
import streamlit as st

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

# Streamlit app setup
st.set_page_config(page_title="RAG-Assisted Claude Chat", layout="wide")
st.title("RAG-Assisted Claude Chat")
st.markdown("A conversational AI powered by RAG-assisted Claude.")

# Function to query Pinecone for relevant context
def retrieve_context(prompt):
    query_embedding = voyage.embed([prompt], model="voyage-3", input_type="query").embeddings[0]
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return "\n\n".join([item["metadata"].get("content", "") for item in results["matches"]])

# Function to fetch Claude's response
def fetch_claude_response(prompt, context, max_tokens=1000):
    structured_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAssistant:"
    response = claude.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        system="You are an expert coder specialized in the boardgame.io library",  # Set the system prompt here
        messages=[
            {"role": "user", "content": structured_prompt}
        ]
    )
    return response["completion"]  # Extract the completion text

# Main Streamlit app logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box with submit on "Enter" and "Shift+Enter" for new line
st.markdown("""
<style>
textarea {
    height: 5em !important;
    resize: none !important;
}
</style>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Type your question below (Shift+Enter for a new line, Enter to submit):",
    value="",
    placeholder="Your question here...",
    on_change=lambda: st.session_state.setdefault("user_submitted", True),
    key="user_input",
)

if st.button("Send") or st.session_state.get("user_submitted", False):
    st.session_state.user_submitted = False
    if user_input.strip():
        context = retrieve_context(user_input)
        response = fetch_claude_response(user_input, context)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Claude", response))
        st.experimental_rerun()  # Refresh the page to display the updated history

# Chat display
for speaker, message in st.session_state.chat_history:
    if speaker == "Claude" and "```" in message:
        st.markdown(f"**{speaker}:**")
        st.code(message.replace("```", ""), language="python")
    else:
        st.markdown(f"**{speaker}:** {message}")

# Scroll the chat history to the top
st.write("")