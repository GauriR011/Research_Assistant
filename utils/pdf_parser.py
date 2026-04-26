import fitz  # PyMuPDF  
# (library to extract raw text from PDFs)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""

    for page in doc: # iterating through the pages in the document
        text += page.get_text() # extracting and appending the text

    return text