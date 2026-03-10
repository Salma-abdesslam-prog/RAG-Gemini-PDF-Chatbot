import re
from sentence_transformers import SentenceTransformer
import numpy as np


def text_process(
    extracted_pages,
    chunk_size=200,
    overlap=50,
    lower=True,
    apply_corrections=False,
    create_vectorstore=True,
    persist_directory=None,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    collection_name="document_embeddings",
):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    # Step 1: Concatenate all page text.
    if isinstance(extracted_pages, list):
        extracted_text = " ".join(page["text"] for page in extracted_pages)
    else:
        extracted_text = str(extracted_pages)

    # Step 2: Basic cleanup.
    extracted_text = re.sub(r"\s+", " ", extracted_text).strip()
    if lower:
        extracted_text = extracted_text.lower()

    # Step 3: Optional OCR corrections.
    if apply_corrections:
        corrections = {
            "0": "o",
            "1": "l",
            "|": "l",
        }
        for wrong, right in corrections.items():
            extracted_text = extracted_text.replace(wrong, right)

    # Step 4: Split into overlapping chunks.
    words = extracted_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    if not chunks:
        raise ValueError("Processed text does not contain any usable chunk.")

    # Step 5: Build embeddings.
    print(f"[INFO] Creating embeddings with model {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings, dtype=float)

    # Step 6: Build a local in-memory vector store (Python 3.14 compatible).
    vectorstore = None
    if create_vectorstore:
        vectorstore = {
            "name": collection_name,
            "documents": chunks,
            "embeddings": embeddings,
            "ids": [str(i) for i in range(len(chunks))],
            "persist_directory": persist_directory,
        }

    print("[INFO] Text processing pipeline completed successfully")
    return {
        "chunks": chunks,
        "embeddings": embeddings.tolist(),
        "vectorstore": vectorstore,
    }
