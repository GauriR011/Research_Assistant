# increasing the chunk size to reduce the number of chunks
def chunk_text(text, 
    chunk_size=1200, 
    overlap=200): # overlap to avoid missing context

    chunks = []
    start = 0 # setting pointer to the beginning of the text

    while start < len(text):
        end = start + chunk_size # calculate the end pointer positon
        chunk = text[start:end] # text chunking 
        chunks.append(chunk)
        start += (chunk_size - overlap) # 

    return chunks