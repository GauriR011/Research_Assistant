import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, texts, metadatas):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, k=4):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx]
            })

        return results