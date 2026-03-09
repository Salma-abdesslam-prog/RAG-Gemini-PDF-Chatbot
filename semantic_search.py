import numpy as np


def semantic_search_chroma(query_embedding, collection, n_results=3):
    """
    Recherche semantique dans un vectorstore local.

    Args:
        query_embedding (list[float] | np.ndarray): embedding du texte a rechercher.
        collection (dict): vectorstore contenant documents/embeddings/ids.
        n_results (int): nombre de resultats similaires a retourner.

    Returns:
        dict: {
            "documents": liste des documents/chunks les plus similaires,
            "ids": liste des identifiants correspondants,
            "distances": liste des distances (1 - similarite cosinus)
        }
    """

    if collection is None:
        raise ValueError("La collection n'est pas chargee.")

    if not isinstance(collection, dict):
        raise TypeError("Format de collection invalide.")

    if query_embedding is None or (isinstance(query_embedding, list) and len(query_embedding) == 0):
        raise ValueError("query_embedding est vide.")

    query = np.array(query_embedding, dtype=float)
    if query.ndim > 1:
        query = query[0]

    docs = collection.get("documents", [])
    ids = collection.get("ids", [])
    emb = collection.get("embeddings")

    if emb is None or len(docs) == 0:
        return {"documents": [], "ids": [], "distances": []}

    emb = np.array(emb, dtype=float)

    # Similarite cosinus stable numeriquement
    q_norm = np.linalg.norm(query)
    e_norm = np.linalg.norm(emb, axis=1)
    denom = (e_norm * q_norm) + 1e-12
    similarities = (emb @ query) / denom

    top_k = max(1, min(n_results, len(docs)))
    top_idx = np.argsort(-similarities)[:top_k]

    top_docs = [docs[i] for i in top_idx]
    top_ids = [ids[i] if i < len(ids) else str(i) for i in top_idx]
    top_distances = [float(1.0 - similarities[i]) for i in top_idx]

    return {
        "documents": top_docs,
        "ids": top_ids,
        "distances": top_distances
    }
