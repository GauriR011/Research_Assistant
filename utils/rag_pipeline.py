import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = genai.GenerativeModel("gemini-1.5-flash")

PROMPT_TEMPLATE = """You are a research assistant helping with literature review.

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

    response = MODEL.generate_content(prompt)

    return response.text