import chromadb
import numpy as np

def semantic_search_chroma(query_embedding, collection, n_results=3):
    """
    Recherche sémantique dans une collection Chroma.

    Args:
        query_embedding (list[float] | np.ndarray): embedding du texte à rechercher
        collection (chromadb.api.models.Collection.Collection): collection Chroma
        n_results (int): nombre de résultats similaires à retourner

    Returns:
        dict: {
            "documents": liste des documents/chunks les plus similaires,
            "ids": liste des identifiants correspondants,
            "distances": liste des distances de similarité (plus petit = plus proche)
        }
    """

    if collection is None:
        raise ValueError("La collection Chroma n'est pas chargée.")

    # Si query_embedding est une liste de vecteurs, on prend le premier
    if isinstance(query_embedding, list) and isinstance(query_embedding[0], (list, np.ndarray)):
        query_embedding = query_embedding[0]


    # Requête dans la collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Retour des résultats
    return {
        "documents": results["documents"][0],
        "ids": results["ids"][0],
        "distances": results["distances"][0]
    }
