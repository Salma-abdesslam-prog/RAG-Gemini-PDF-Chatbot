import os
from typing import Dict, List, Any
from google import genai
from google.genai.errors import APIError
from google.genai import types

def generate_rag_answer_from_search(
    search_output: Dict[str, List[str]],
    user_prompt: str,
    llm_model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    max_output_tokens: int = 512
) -> str:
    """
    Génère une réponse RAG (Retrieval-Augmented Generation) à partir de l'output
    de la recherche sémantique, en utilisant l'API Gemini 2.5 Flash.

    L'authentification utilise la variable d'environnement GEMINI_API_KEY.

    search_output doit être un dictionnaire du type :
        {
            "documents": [...],   # liste de chunks textuels (str)
            "ids": [...],
            "distances": [...]
        }
    """

    # --- 1. Initialisation du client Gemini ---
    # Le client récupère automatiquement la clé API de l'environnement GEMINI_API_KEY
    try:
        client = genai.Client()
    except Exception:
        # Gérer le cas où la clé n'est pas définie ou est invalide.
        return "❌ Erreur: GEMINI_API_KEY n'est pas définie ou est invalide."


    # --- 2. Vérifications minimales et préparation du contexte ---
    if not search_output or "documents" not in search_output or not isinstance(search_output.get("documents"), list):
        return "❌ search_output invalide : clé 'documents' manquante ou format incorrect."

    context_chunks = search_output["documents"]
    if len(context_chunks) == 0:
        return "❌ Aucun chunk de contexte trouvé par la recherche sémantique."

    # Construction du contexte complet, séparé par des doubles sauts de ligne
    context_text = "\n\n".join(context_chunks)

    # --- 3. Création du prompt RAG (Instructions + Contexte + Question) ---
    system_instruction = (
        "Tu es un assistant intelligent et serviable qui répond **uniquement** à partir du contexte "
        "fourni. Si la réponse ne se trouve pas dans le contexte, tu dois répondre "
        "poliment que l'information n'est pas disponible dans les sources fournies. "
        "Tes réponses doivent être concises et directes."
    )

    full_prompt = f"""
=== CONTEXTE FOURNI ===
{context_text}

=== QUESTION DE L'UTILISATEUR ===
{user_prompt}
"""

    # --- 4. Configuration de la génération ---
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    # --- 5. Appel à l'API Gemini ---
    try:
        response = client.models.generate_content(
            model=llm_model,
            contents=full_prompt,
            config=config
        )

        # Vérifier si une réponse valide a été générée
        if response.text:
            return response.text.strip()
        else:
            return "⚠️ Le modèle a généré une réponse vide ou bloquée."

    except APIError as e:
        return f"❌ Erreur API Gemini ({e.status_code}) : {e.message}"
    except Exception as e:
        return f"❌ Erreur inattendue lors de l'appel à Gemini : {e}"

# ----------------------------------------------------------------------
# --- EXEMPLE D'UTILISATION (Pour tester la fonction) ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Assurez-vous que la variable GEMINI_API_KEY est définie dans votre environnement
    if "GEMINI_API_KEY" not in os.environ:
        print("🚨 VEUILLEZ DÉFINIR LA VARIABLE D'ENVIRONNEMENT 'GEMINI_API_KEY'")
    else:
        # Simulation de l'output d'une recherche sémantique (RAG)
        simulated_search_output: Dict[str, List[str]] = {
            "documents": [
                "Le langage Python, créé par Guido van Rossum, est sorti pour la première fois en 1991. Il est très apprécié pour sa lisibilité et sa syntaxe claire.",
                "Gemini 2.5 Flash est un modèle multimodal conçu pour la rapidité et l'efficacité, avec une fenêtre contextuelle d'un million de jetons."
            ],
            "ids": ["doc_1", "doc_2"],
            "distances": [0.15, 0.22]
        }
        
        user_question = "Qui est le créateur du langage Python et quand a-t-il été publié ?"

        print(f"Question : {user_question}")
        print("Génération de la réponse RAG avec Gemini 2.5 Flash...")
        
        answer = generate_rag_answer_from_search(
            search_output=simulated_search_output,
            user_prompt=user_question,
            temperature=0.1 # Température basse pour une réponse plus factuelle/déterminée
        )
        
        print("\n=== RÉPONSE FINALE DU MODÈLE ===")
        print(answer)
        print("================================")