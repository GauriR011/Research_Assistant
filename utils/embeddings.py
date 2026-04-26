import os
import google.genai as genai
from dotenv import load_dotenv
import tqdm
import streamlit as st


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_model = "models/gemini-embedding-001"

# converting text chunks into numerical vectors
# we are using batching to increase efficiency and avoid hitting API limits
def get_embeddings(texts, batch_size=20):
    all_embeddings = []

    # for the progress bar
    progress = st.progress(0)
    total_batches = len(texts) // batch_size + 1

    # iterating through the batches (sectioning the text into batches by using the batch size as our step size)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # send the batch object to the LLM to embed the text
        response = client.models.embed_content(
            model=_model,
            contents=batch
        )
        
        # extracting the numerical vectors from the LLM response
        batch_embeddings = [e.values for e in response.embeddings]

        if len(batch_embeddings) != len(batch): # safety check
        # if the number of embeddings is different from the number of batches, the an error is thown
            raise ValueError(
                f"Embedding mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
            )

        # adding the extracted embeddings to the embeddings list
        all_embeddings.extend(batch_embeddings)

        # updating the Streamlit progress bar
        progress.progress((i // batch_size + 1) / total_batches)
    
    return all_embeddings


# converts the user's question into embeddings
def embed_query(query):
    response = client.models.embed_content(
        model=_model,
        contents=query
    )
    return response.embeddings[0].values # return the first numeric vector from the response (since there is only 1 question)


# so how this works is that algorithm converts the user question into vector embeddings
# it them compares the question embeddings with the text embeddings
# It then calculates which document embeddings are "mathematically closest" to the question embeddings.