# app.py (Streamlit - Corrigé pour Gemini RAG)
import os
import json
import tempfile
import streamlit as st

# Importez vos modules locaux
from Text_processing import text_process
from OCR import pdf_to_text_json
from semantic_search import semantic_search_chroma
from sentence_transformers import SentenceTransformer

# --- CHANGEMENT 1 : Importer la nouvelle fonction Gemini RAG ---
# Assurez-vous que le fichier generate_answer_gemini.py est bien dans votre dossier
from generate_answer import generate_rag_answer_from_search 

# --- CONFIGURATION INITIALE & TITRE ---
st.set_page_config(layout="wide")
st.title("Chat with your PDF via Gemini 🤖")

# --- CHANGEMENT 2 : Vérification de la clé Gemini ---
# Le SDK 'google-genai' lit automatiquement GEMINI_API_KEY.
# On vérifie si la clé est présente pour guider l'utilisateur.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Optionnel : Permettre la saisie de la clé dans l'interface Streamlit
    GEMINI_API_KEY = st.text_input(
        "Entrez votre **GEMINI_API_KEY** :", 
        type="password",
        help="La clé est lue par la variable d'environnement GEMINI_API_KEY."
    )
    if not GEMINI_API_KEY:
        st.error("⚠️ Clé Gemini API requise. Veuillez la définir dans l'environnement ou ci-dessus.")
        # Arrêter l'exécution de l'application si la clé manque
        st.stop()
else:
    st.success("🔑 Authentification Gemini API prête.")


# --- INTERFACE UTILISATEUR ---
uploaded_file = st.file_uploader("Déposez votre fichier PDF ici :", type=["pdf"])
user_prompt = st.chat_input("Posez votre question ici...")


# --- LOGIQUE PRINCIPALE ---
if uploaded_file is not None:
    # Affichage de confirmation de fichier
    st.info(f"Fichier reçu : {uploaded_file.name}. Traitement en cours...")

    # Sauvegarde temporaire du PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Extraction texte via ton OCR
        with st.spinner("1/3 - Extraction du texte..."):
            temp_json_path = pdf_to_text_json(tmp_path)
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                extracted_text = json.load(f)

        # Création des chunks et du vectorstore (Chroma)
        with st.spinner("2/3 - Création des chunks et de la base vectorielle..."):
            pdf_data = text_process(
                extracted_text,
                chunk_size=200,
                overlap=50,
                lower=True,
                apply_corrections=False,
                create_vectorstore=True,
                persist_directory=None,
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            collection = pdf_data["vectorstore"]

        # Si l'utilisateur a posé une question et que le traitement est terminé
        if user_prompt:
            st.info("3/3 - Recherche sémantique en cours...")
            
            # Encoder la question
            embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            query_embedding = embedder.encode(user_prompt).tolist()

            # Appel de la recherche sémantique
            search_output = semantic_search_chroma(query_embedding, collection, n_results=3)

            # Afficher les passages utilisés (transparence)
            with st.expander("🔍 Passages de contexte trouvés :"):
                for i, chunk in enumerate(search_output.get("documents", []), start=1):
                    st.markdown(f"**Passage {i} (Distance: {search_output.get('distances', [None]*i)[i-1]:.4f}):**")
                    st.write(chunk)

            # Génération RAG avec Gemini 2.5 Flash
            with st.spinner("🧠 Génération de la réponse avec Gemini 2.5 Flash..."):
                answer = generate_rag_answer_from_search(
                    search_output=search_output,
                    user_prompt=user_prompt,
                    llm_model="gemini-2.5-flash" # Spécification du modèle Gemini
                )

            st.subheader("💬 Réponse générée :")
            st.success(answer)
            
    finally:
        # Nettoyage des fichiers temporaires
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if 'temp_json_path' in locals() and os.path.exists(temp_json_path):
            os.remove(temp_json_path)