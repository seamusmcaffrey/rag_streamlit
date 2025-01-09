import os
import streamlit as st
import anthropic
import voyageai
from pinecone import Pinecone

# -----------------------------------------------------------------------------
# 1. Retrieve and store your API keys from the environment (or however you load them)
# -----------------------------------------------------------------------------
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")

if not VOYAGE_API_KEY or not CLAUDE_API_KEY or not PINECONE_API_KEY:
    st.error("Missing one or more required API keys. Please set VOYAGE_API_KEY, CLAUDE_API_KEY, and PINECONE_API_KEY.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Initialize Voyage and Pinecone clients
# -----------------------------------------------------------------------------
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "rag-index"
index = pc.Index(INDEX_NAME)

# -----------------------------------------------------------------------------
# 3. Initialize Anthropic client (messages-based)
# -----------------------------------------------------------------------------
claude = anthropic.Client(api_key=CLAUDE_API_KEY)

# -----------------------------------------------------------------------------
# 4. Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RAG-Assisted Claude Chat", layout="wide")
st.title("RAG-Assisted Claude Chat")
st.markdown("A conversational AI powered by RAG-assisted Claude.")

# -----------------------------------------------------------------------------
# 5. Context retrieval from Pinecone
# -----------------------------------------------------------------------------
def retrieve_context(query_text):
    """
    Takes the user's query, embeds it, and retrieves top_k matching docs.
    Returns a string of relevant context combined.
    """
    try:
        query_embedding = voyage.embed([query_text], model="voyage-3", input_type="query").embeddings[0]
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        # Combine the 'content' fields from all matches
        combined_context = "\n\n".join([match["metadata"].get("content", "") for match in results["matches"]])
        return combined_context
    except Exception as e:
        st.warning(f"Error retrieving context from Pinecone: {e}")
        return ""

# -----------------------------------------------------------------------------
# 6. Function to call Claude (Anthropic messages-based API)
# -----------------------------------------------------------------------------
def fetch_claude_response(user_query, context, max_tokens=1000):
    """
    Uses Anthropic's Chat-based endpoint with the messages param.
    Returns a string (Claude's completion).
    """
    try:
        # Build your conversation as a list of messages
        # We'll use a "system" role to set instructions for Claude,
        # plus a "user" role to pass the combined RAG context + user question.
        # For additional prompt engineering, refine these messages as you wish.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert coder specialized in the boardgame.io library. "
                    "Use the provided context to answer the user's question. If the "
                    "context is insufficient, say you are unsure."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{user_query}\n\n"
                    "Please provide a helpful, code-focused answer where applicable."
                )
            }
        ]

        # Send the request to Anthropic
        response = claude.chat.create(
            model="claude-3-5-sonnet-20241022",           # or 'claude-instant-1' if needed
            messages=messages,
            max_tokens_to_sample=max_tokens,
            temperature=0.7,
            top_p=1
        )

        # The actual text of Claude's reply is in response["completion"]
        raw_reply = response["completion"]  # string
        # Optionally remove weird artifacts if you have any
        cleaned_reply = raw_reply.replace("[TextBlock(text=", "").replace(", type='text')]", "").strip()
        return cleaned_reply

    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return "Sorry, there was an error."

# -----------------------------------------------------------------------------
# 7. Setup Streamlit session to store chat history
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -----------------------------------------------------------------------------
# 8. Main user input + button
# -----------------------------------------------------------------------------
st.subheader("Ask a question about boardgame.io:")
user_input = st.text_area("Your question here:", value="", height=100)

if st.button("Submit"):
    # Only proceed if user typed something
    if user_input.strip():
        # 1) Add user message to the chat history
        st.session_state.chat_history.append(("User", user_input.strip()))
        # 2) Retrieve context from Pinecone
        context_text = retrieve_context(user_input.strip())
        # 3) Call Claude for the response
        claude_reply = fetch_claude_response(user_input.strip(), context_text)
        # 4) Append Claude's response to chat history
        st.session_state.chat_history.append(("Claude", claude_reply))

# -----------------------------------------------------------------------------
# 9. Render the conversation
# -----------------------------------------------------------------------------
st.write("---")
st.subheader("Conversation History")

for speaker, msg in st.session_state.chat_history:
    if speaker == "Claude":
        # If there's a code fence (```), optionally render it as st.code
        # Otherwise just render as text
        if "```" in msg:
            # A simple approach: remove backticks and display
            # Alternatively, parse code blocks in more detail if needed
            code_snippet = msg.replace("```", "")
            st.markdown(f"**{speaker}:**")
            st.code(code_snippet, language="python")
        else:
            st.markdown(f"**{speaker}:** {msg}")
    else:
        st.markdown(f"**{speaker}:** {msg}")