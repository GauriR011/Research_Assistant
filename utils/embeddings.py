import os
import google.genai as genai
from dotenv import load_dotenv
import tqdm
import streamlit as st


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_model = "models/gemini-embedding-001"

def get_embeddings(texts, batch_size=20):
    all_embeddings = []
    progress = st.progress(0)

    total_batches = len(texts) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        response = client.models.embed_content(
            model=_model,
            contents=batch
        )

        batch_embeddings = [e.values for e in response.embeddings]

        if len(batch_embeddings) != len(batch):
            raise ValueError(
                f"Embedding mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
            )

        all_embeddings.extend(batch_embeddings)

        progress.progress((i // batch_size + 1) / total_batches)
    
    return all_embeddings


def embed_query(query):
    response = client.models.embed_content(
        model=_model,
        contents=query
    )
    return response.embeddings[0].values