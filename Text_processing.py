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
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    collection_name="document_embeddings"
):
    if chunk_size <= overlap:
        raise ValueError("chunk_size doit etre strictement superieur a overlap.")

    # Etape 1 : Concatenation
    if isinstance(extracted_pages, list):
        extracted_text = " ".join(page['text'] for page in extracted_pages)
    else:
        extracted_text = str(extracted_pages)

    # Etape 2 : Nettoyage
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
    if lower:
        extracted_text = extracted_text.lower()

    # Etape 3 : Corrections OCR si necessaire
    if apply_corrections:
        corrections = {
            '0': 'o',
            '1': 'l',
            '|': 'l',
        }
        for wrong, right in corrections.items():
            extracted_text = extracted_text.replace(wrong, right)

    # Etape 4 : Decoupage en chunks
    words = extracted_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    if not chunks:
        raise ValueError("Le texte traite ne contient aucun chunk utilisable.")

    # Etape 5 : Embeddings
    print(f"[INFO] Creation des embeddings avec le modele {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings, dtype=float)

    # Etape 6 : Vectorstore local (compatible Python 3.14)
    vectorstore = None
    if create_vectorstore:
        vectorstore = {
            "name": collection_name,
            "documents": chunks,
            "embeddings": embeddings,
            "ids": [str(i) for i in range(len(chunks))],
            "persist_directory": persist_directory,
        }

    print("[INFO] Pipeline de traitement termine avec succes")
    return {
        "chunks": chunks,
        "embeddings": embeddings.tolist(),
        "vectorstore": vectorstore,
    }
