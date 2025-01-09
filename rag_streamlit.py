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
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": structured_prompt}
        ]
    )
    return response.content  # Streamlit automatically handles JSON and string outputs.

# Main Streamlit app logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Your Question:", "")
if st.button("Send"):
    if user_input:
        context = retrieve_context(user_input)
        response = fetch_claude_response(user_input, context)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Claude", response))
        user_input = ""  # Clear input field

# Display chat history
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")