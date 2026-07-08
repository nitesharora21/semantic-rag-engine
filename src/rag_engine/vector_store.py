import faiss
import numpy as np


class FaissVectorStore:
    """
    Simple FAISS vector store for semantic search.

    This version builds an in-memory index from embeddings.
    It does not persist FAISS index yet.
    """

    def __init__(self, embeddings: list[list[float]]) -> None:
        if not embeddings:
            raise ValueError("embeddings must not be empty")

        self.embeddings = np.array(embeddings, dtype="float32")
        # shape = # of chunks X length of each embedding
        self.dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[tuple[float, int]]:
        """
        Search FAISS index using a query embedding.

        Returns:
          A list of tuples of the type [float, int]
          (score, index)
        In here - the index points to the original chunk position.
        """
        # Convert the vector embedding to np.array()
        query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(x=query_vector, k=top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append((float(score), int(index)))

        return results
