import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT_TEMPLATE = """You are a research assistant helping with literature review and methodolody of the research papers.

Instructions:
- Answer ONLY using the provided context
- If multiple documents are relevant, compare them
- Cite the document name for every claim
- If the answer is not in the context, say 'Not enough information'

Context:
{retrieved_chunks}

Question:
{user_query}
"""

def format_context(chunks):
    context = ""
    for chunk in chunks:
        context += f"{chunk['text']}\n(Source: {chunk['metadata']['source']})\n\n"
    return context


def generate_answer(chunks, query):
    context = format_context(chunks)

    prompt = PROMPT_TEMPLATE.format(
        retrieved_chunks=context,
        user_query=query
    )

    response = client.models.generate_content(
        model="gemini-flash-latest", # using the latest gemini flash model
        contents=prompt
    )

    return response.text