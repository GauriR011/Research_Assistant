import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT_TEMPLATE = """
You are a research assistant helping with literature review and methodolody of the research papers.

Instructions:
- Answer ONLY using the provided context
- If multiple documents are relevant, compare them
- Write the answers in bullet points and give appropriate headers to sections, wherever necessary
- Use citations in the format (Source 1), (Source 2) and apply bold formatting to them
- Do NOT use square bracket citations like [1] to avoid confusion with paper citations
- Each reference corresponds to a document in the reference list
- If the answer is not in the context, say 'Not enough information'

Context:
{retrieved_chunks}

Question:
{user_query}
"""

# def format_context(chunks):
#     context = ""
#     for chunk in chunks:
#         context += f"{chunk['text']}\n(Source: {chunk['metadata']['source']})\n\n"
#     return context

# def format_context(chunks):
#     context = ""
#     source_map = {}
#     source_counter = 1

#     for chunk in chunks:
#         source = chunk["metadata"]["source"]

#         # Assign number if new source
#         if source not in source_map:
#             source_map[source] = source_counter
#             source_counter += 1

#         ref_id = source_map[source]

#         context += f"{chunk['text']}\n(Ref {ref_id})\n\n"

#     return context, source_map


def format_context(chunks):
    context = ""

    for chunk in chunks:
        ref_id = chunk["ref_id"]
        context += f"{chunk['text']}\n(Ref {ref_id})\n\n"

    return context


# def generate_answer(chunks, query):
#     context = format_context(chunks)

#     prompt = PROMPT_TEMPLATE.format(
#         retrieved_chunks=context,
#         user_query=query
#     )

#     response = client.models.generate_content(
#         model="gemini-flash-latest", # using the latest gemini flash model
#         contents=prompt
#     )

#     return response.text

# def generate_answer(chunks, query):
#     context, source_map = format_context(chunks)

#     prompt = PROMPT_TEMPLATE.format(
#         retrieved_chunks=context,
#         user_query=query
#     )

#     response = client.models.generate_content(
#         model="gemini-flash-latest", # using the latest gemini flash model
#         contents=prompt
#     )

#     return response.text, source_map

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