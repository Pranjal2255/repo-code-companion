import os
import pickle
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# Browser Tab Interface Styling Configuration
st.set_page_config(page_title="Git Repo Companion", page_icon="💻", layout="wide")

@st.cache_resource
def load_code_resources():
    """Loads the compiled vector database and model from disk into cache memory once."""
    if not os.path.exists("code_vector_index.bin") or not os.path.exists("code_metadata.pkl"):
        return None, None, None
    
    index = faiss.read_index("code_vector_index.bin")
    with open("code_metadata.pkl", "rb") as f:
        data = pickle.load(f)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, data["chunks"], model

# Instantiate dependencies
index, chunks, embedding_model = load_code_resources()

# UI Layout Header
st.title("💻 Open-Source Git Repository Code Companion")
st.subheader("Query codebase architecture, files, and functions completely offline")
st.markdown("---")

if index is None:
    st.error("❌ Vector database objects missing! Run `python ingest.py` in your terminal environment first.")
else:
    # Sidebar Metrics Context
    with st.sidebar:
        st.header("⚙️ Codebase Analytics")
        st.success(f"Successfully Indexed: {len(chunks)} Functional Blocks")
        st.info("Target Repository: pallets/click")
        st.markdown("This assistant is context-grounded. It uses structured functions retrieved from local index components to construct its responses.")

    # Handle Streamlit persistent execution state arrays
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display running chat entries
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Live Input Interface Capture
    if query_input := st.chat_input("Ask a question about the repository code logic..."):
        
        # Stream User Entry to Screen
        with st.chat_message("user"):
            st.markdown(query_input)
        st.session_state.chat_history.append({"role": "user", "content": query_input})

        # Process inference
        with st.chat_message("assistant"):
            output_frame = st.empty()
            output_frame.markdown("*Scanning codebase functional geometry & querying local LLM...*")

            # 1. Vectorize query string
            query_vector = embedding_model.encode([query_input])[0][None, :]

            # 2. Extract Top 3 closest structural code matches (k=3)
            distances, indices = index.search(query_vector, k=3)

            # 3. Pull matches from mapping dictionaries
            retrieved_code_blocks = []
            for idx in indices[0]:
                if idx != -1 and idx < len(chunks):
                    retrieved_code_blocks.append(chunks[idx])
            
            context_payload = "\n\n# ==========================================\n\n".join(retrieved_code_blocks)

            # 4. Construct System Instructions with strict context boundaries
            system_prompt = (
                "You are an expert principal software architect analyzing a Python codebase.\n"
                "Answer the user's technical query using ONLY the provided code snippets in the context block below.\n"
                "Reference specific filenames and functions when explaining logic. If the answer cannot be confidently deduced "
                "solely from the context block, state 'I cannot verify this logic loop inside the indexed source files.'\n\n"
                f"Codebase Context:\n{context_payload}"
            )

            # 5. Interface Payload via local Ollama Engine
            try:
                response = ollama.generate(
                    model="llama3",
                    system=system_prompt,
                    prompt=query_input
                )
                execution_answer = response['response']
                output_frame.markdown(execution_answer)
                
                # Expandable Code Lineage inspector panel for data tracking transparency
                with st.expander("🔍 Inspect Extracted Functional Code Lineage"):
                    for i, block in enumerate(retrieved_code_blocks):
                        st.markdown(f"**Retrieved Structure Component [{i+1}]:**")
                        st.code(block, language="python")
                        
            except Exception as error_msg:
                execution_answer = f"Ollama execution daemon timeout: {str(error_msg)}"
                output_frame.error(execution_answer)

            # Update session history states
            st.session_state.chat_history.append({"role": "assistant", "content": execution_answer})