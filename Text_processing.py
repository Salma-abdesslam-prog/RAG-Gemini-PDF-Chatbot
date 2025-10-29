import re
import chromadb
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
    model_name='sentence-transformers/all-MiniLM-L6-v2'
):
    # Étape 1 : Concaténation
    if isinstance(extracted_pages, list):
        extracted_text = " ".join(page['text'] for page in extracted_pages)
    else:
        extracted_text = str(extracted_pages)

    # Étape 2 : Nettoyage
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
    if lower:
        extracted_text = extracted_text.lower()

    # Étape 3 : Corrections OCR si nécessaire
    if apply_corrections:
        corrections = {
            '0': 'o', '1': 'l', '|': 'l', '“': '"', '”': '"', '‘': "'", '’': "'"
        }
        for wrong, right in corrections.items():
            extracted_text = extracted_text.replace(wrong, right)

    # Étape 4 : Découpage en chunks
    words = extracted_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    if not chunks:
        raise ValueError("Le texte traité ne contient aucun chunk utilisable.")

    # Étape 5 : Embeddings
    print(f"[INFO] Création des embeddings avec le modèle {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings, dtype=float).tolist()  

    # Étape 6 : Vectorstore Chroma (optionnel)
    vectorstore = None
    if create_vectorstore:
        print("[INFO] Création du vectorstore Chroma...")
        client = chromadb.Client()

        if persist_directory:
            collection = client.get_or_create_collection(
                name="document_embeddings",
                metadata={"persist_directory": persist_directory}
            )
        else:
            collection = client.get_or_create_collection(name="document_embeddings")

        for i, embedding in enumerate(embeddings):
            collection.add(
                documents=[chunks[i]],
                embeddings=[embedding],  
                ids=[str(i)]
            )

        vectorstore = collection

    print("[INFO] Pipeline de traitement terminé avec succès ")
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "vectorstore": vectorstore
    }





# # Exemple d'utilisation :
# temp_json_path = pdf_to_text_json('/home/salma/Downloads/Salma-Abdesslam CV-1.pdf')
# print(f"Le fichier temporaire JSON est créé à : {temp_json_path}")
# with open(temp_json_path, 'r', encoding='utf-8') as f:
#     extracted_text = json.load(f)
#     chunks = preprocess_and_chunk(extracted_text)
# print(chunks)    
# vector = embedding_creation(chunks)
# print(vector)
# vectorstore_creation(vector)