import os
import google.generativeai as genai
from dotenv import load_dotenv
import tqdm
import streamlit as st


load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_model = "models/gemini-embedding-001"

def get_embeddings(texts, batch_size=20):
    all_embeddings = []
    progress = st.progress(0)

    total_batches = len(texts) // batch_size + 1

    # adding a progress bar for pdf processing
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        response = genai.embed_content(
            model=_model,
            content=batch
        )

        all_embeddings.extend(response["embedding"])

        progress.progress((i // batch_size + 1) / total_batches)
    
    return all_embeddings


def embed_query(query):
    response = genai.embed_content(
    model=_model,
        content=query
    )
    return response["embedding"]