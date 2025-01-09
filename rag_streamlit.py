import os
import anthropic
import voyageai
from pinecone import Pinecone
import streamlit as st

# Set your API keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize clients
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "rag-index"
index = pc.Index(INDEX_NAME)

# Anthropic client
claude = anthropic.Client(api_key=CLAUDE_API_KEY)

# Streamlit app setup
st.set_page_config(page_title="RAG-Assisted Claude Chat", layout="wide")
st.title("RAG-Assisted Claude Chat")
st.markdown("A conversational AI powered by RAG-assisted Claude.")

# 1. Pinecone retrieval
def retrieve_context(prompt):
    query_embedding = voyage.embed([prompt], model="voyage-3", input_type="query").embeddings[0]
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return "\n\n".join([item["metadata"].get("content", "") for item in results["matches"]])

# 2. Fetch Claude response via Anthropic Completions
def fetch_claude_response(prompt, context, max_tokens=1000):
    # Build a prompt with the recommended Anthropic approach
    full_prompt = (
        f"{anthropic.HUMAN_PROMPT} "
        f"You are an expert coder specialized in the boardgame.io library.\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{prompt}\n"
        f"{anthropic.AI_PROMPT}"
    )
    
    # Completions call
    response = claude.completions.create(
        prompt=full_prompt,
        model="claude-3-5-sonnet-20241022",               # or whichever variant
        max_tokens_to_sample=max_tokens,
        temperature=0.7
        # anthropic docs: https://github.com/anthropic/anthropic-sdk-python
    )
    
    # Extract the text
    raw_text = response["completion"]
    
    # Clean extraneous artifacts if any
    clean_response = raw_text.replace("[TextBlock(text=", "").replace(", type='text')]", "").strip()
    return clean_response

# 3. Streamlit session for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def submit_message():
    user_input = st.session_state["user_input"].strip()
    if user_input:
        # Add to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state["user_input"] = ""

        # Retrieve relevant context, then ask Claude
        context = retrieve_context(user_input)
        cl_reply = fetch_claude_response(user_input, context)
        st.session_state.chat_history.append(("Claude", cl_reply))

# 4. Text input with Enter to submit, Shift+Enter for new line
st.text_area(
    "Type your message here (Enter = submit, Shift+Enter = new line):",
    key="user_input",
    on_change=submit_message,
    placeholder="Enter your message...",
    height=100
)

# 5. Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "Claude":
        st.markdown(f"**{speaker}:** {message}")
    else:
        st.markdown(f"**{speaker}:** {message}")