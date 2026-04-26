import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

# initializing the Gemini API key
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

# Adding inline refereces/sources into the text genreated by the LLM
def format_context(chunks):
    context = ""

    for chunk in chunks:
        # reference id of the chunk (tells the LLM which document did the text come from)
        ref_id = chunk["ref_id"] 
        # adding the reference to text
        context += f"{chunk['text']}\n(Ref {ref_id})\n\n"

    return context



def generate_answer(chunks, query):
    context = format_context(chunks)


    # injecting user query and geneated context into the template 
    prompt = PROMPT_TEMPLATE.format(
        retrieved_chunks=context,
        user_query=query
    )

    # LMM response object
    response = client.models.generate_content(
        model= "gemini-flash-latest",# using the latest gemini flash model
        contents=prompt
    )

    # return text component of the LMM response object
    return response.text