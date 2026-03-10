import numpy as np


def semantic_search_chroma(query_embedding, collection, n_results=3):
    """
    Run semantic search on a local vector store.

    Args:
        query_embedding (list[float] | np.ndarray): query embedding.
        collection (dict): vector store containing documents/embeddings/ids.
        n_results (int): number of similar results to return.

    Returns:
        dict: {
            "documents": most similar chunks,
            "ids": matching ids,
            "distances": distance values (1 - cosine similarity)
        }
    """

    if collection is None:
        raise ValueError("Collection is not loaded.")

    if not isinstance(collection, dict):
        raise TypeError("Invalid collection format.")

    if query_embedding is None or (isinstance(query_embedding, list) and len(query_embedding) == 0):
        raise ValueError("query_embedding is empty.")

    query = np.array(query_embedding, dtype=float)
    if query.ndim > 1:
        query = query[0]

    docs = collection.get("documents", [])
    ids = collection.get("ids", [])
    emb = collection.get("embeddings")

    if emb is None or len(docs) == 0:
        return {"documents": [], "ids": [], "distances": []}

    emb = np.array(emb, dtype=float)

    # Numerically stable cosine similarity.
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
        "distances": top_distances,
    }
