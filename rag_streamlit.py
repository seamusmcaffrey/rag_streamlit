import os
import json
import voyageai
from pinecone import Pinecone
from anthropic import Client
import streamlit as st

# Set your API keys
VOYAGE_API_KEY = "pa-igkhGLYmF_UOg6yhLHQm9-vMgfVVzWThHPTS-zIko5Q"
CLAUDE_API_KEY = "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"
PINECONE_API_KEY = "pcsk_3BYduG_UMor22qkDcB6zFC4ZanvU75m8gdMbevXpiTHQGHpCWi1rcZia8hVV85J8on6n3E"
PINECONE_ENV = "us-east-1"

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