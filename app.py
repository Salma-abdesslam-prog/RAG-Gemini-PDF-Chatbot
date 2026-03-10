import os
import json
import tempfile
import hashlib
import streamlit as st
from sentence_transformers import SentenceTransformer

from Text_processing import text_process
from OCR import pdf_to_text_json
from semantic_search import semantic_search_chroma
from generate_answer import generate_rag_answer_from_search


st.set_page_config(layout="wide")
st.title("Chat with your PDF via Gemini")


@st.cache_resource
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def ensure_api_key() -> bool:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        st.success("Gemini API authentication ready.")
        return True

    entered_key = st.text_input(
        "Enter your GEMINI_API_KEY:",
        type="password",
        help="The key is read from the GEMINI_API_KEY environment variable.",
    )

    if not entered_key:
        st.error("Gemini API key is required.")
        return False

    # Make the key available to the google-genai SDK.
    os.environ["GEMINI_API_KEY"] = entered_key
    return True


def build_collection_from_pdf(pdf_bytes: bytes, model_name: str):
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("current_file_hash") == file_hash and st.session_state.get("collection") is not None:
        return st.session_state["collection"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    temp_json_path = None
    try:
        with st.spinner("1/3 - Extracting text..."):
            temp_json_path = pdf_to_text_json(tmp_path)
            with open(temp_json_path, "r", encoding="utf-8") as f:
                extracted_text = json.load(f)

        with st.spinner("2/3 - Building chunks and vector index..."):
            pdf_data = text_process(
                extracted_text,
                chunk_size=200,
                overlap=50,
                lower=True,
                apply_corrections=False,
                create_vectorstore=True,
                persist_directory=None,
                model_name=model_name,
                collection_name=f"document_embeddings_{file_hash[:12]}",
            )

        st.session_state["current_file_hash"] = file_hash
        st.session_state["collection"] = pdf_data["vectorstore"]
        return pdf_data["vectorstore"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if temp_json_path and os.path.exists(temp_json_path):
            os.remove(temp_json_path)


if not ensure_api_key():
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF file:", type=["pdf"])
user_prompt = st.chat_input("Ask your question here...")

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
    st.stop()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

try:
    pdf_bytes = uploaded_file.getvalue()
    collection = build_collection_from_pdf(pdf_bytes, model_name)
except Exception as e:
    st.error("Error while preparing the document.")
    st.exception(e)
    st.stop()

if user_prompt:
    try:
        with st.spinner("3/3 - Running semantic search..."):
            embedder = get_embedder(model_name)
            query_embedding = embedder.encode(user_prompt).tolist()
            search_output = semantic_search_chroma(query_embedding, collection, n_results=3)

        with st.expander("Retrieved context passages:"):
            documents = search_output.get("documents", [])
            distances = search_output.get("distances", [])
            for i, chunk in enumerate(documents, start=1):
                distance_text = "N/A"
                if i - 1 < len(distances) and distances[i - 1] is not None:
                    distance_text = f"{distances[i - 1]:.4f}"
                st.markdown(f"**Passage {i} (Distance: {distance_text})**")
                st.write(chunk)

        with st.spinner("Generating answer with Gemini..."):
            answer = generate_rag_answer_from_search(
                search_output=search_output,
                user_prompt=user_prompt,
                llm_model="gemini-2.5-flash",
            )

        st.subheader("Generated answer:")
        st.success(answer)
    except Exception as e:
        st.error("Error while searching or generating the answer.")
        st.exception(e)
