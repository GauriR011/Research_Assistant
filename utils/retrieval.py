import faiss
import numpy as np
import pickle
import os

# we will be using FAISS (Facebook AI Similarity Search) to see which of the vector embeddings matches the closest
# to the user query vector.

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim) # measures the Euclidean distance between vectors.
        self.texts = []
        self.metadata = []
        self.embeddings = []

    def add(self, embeddings, texts, metadatas): # safety check
        assert len(embeddings) == len(texts) == len(metadatas), \
            f"Mismatch: {len(embeddings)}, {len(texts)}, {len(metadatas)}"

        # converting embeddings into a numpy array using 32-bit floats
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors) # passing the vectors to FAISS

        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, k=4):
        query_vector = np.array([query_embedding]).astype("float32")
        # It finds the top k vectors in the database that are most similar to the user question.
        # and returns their distances and positions in the list (indices)
        distances, indices = self.index.search(query_vector, k)

        results = []

        # we will use these indices to fetch the actual text and metadata from the lists
        for idx in indices[0]:
            if idx == -1 or idx >= len(self.texts):
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
            })

        return results

    def save(self, path):
        # saving the index
        faiss.write_index(self.index, path + ".index")

        # using pickle to save text, metadata files and embeddings  
        with open(path + ".pkl", "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "embeddings": self.embeddings
            }, f)

    @classmethod
    # reconstructing the class from the index and pickled files so we won't need to re-process 
    # the same PDFs every time the app is restarted.
    def load(cls, path):
        # Load FAISS index
        index = faiss.read_index(path + ".index")

        # Load metadata
        with open(path + ".pkl", "rb") as f:
            data = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.texts = data.get("texts", [])
        store.metadata = data.get("metadata", [])

        return store



    # To summarize the, 
    # On reading new data: Text --→ Embedding --→ FAISS Index.
    # Closest chunk matching: User Question --→ Query Embedding --→ FAISS search --→ Top k Text Chunks.