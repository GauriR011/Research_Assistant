import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embeddings(texts):
    embeddings = []

    for text in texts:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        embeddings.append(response["embedding"])

    return embeddings


def embed_query(query):
    response = genai.embed_content(
        model="models/embedding-001",
        content=query
    )
    return response["embedding"]