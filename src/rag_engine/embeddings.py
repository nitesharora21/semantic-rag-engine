from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Small wrapper around a sentence-transformers embedding model.
    Now we will keep the embedding logic in one place, as the rest of the
    project will reuse the class for embedding the vectors.

    Note: For now - using the all-MiniLM-L6-v2 since its recommened as a
    good first embedding model. Its small, fast, and generally used for
    semantic searches.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_text(self, input_text: str) -> list[float]:
        """
        Convert the given text into a vector embedding.
        """
        embedding = self.model.encode(input_text)
        return embedding.tolist()

    def embed_texts(self, input_texts: list[str]) -> list[list[float]]:
        """
        Convert given multiple input_texts into embedding vectors.
        """
        embeddings = self.model.encode(input_texts)
        return embeddings.tolist()
