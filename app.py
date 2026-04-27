import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from utils.embeddings import get_embeddings, embed_query
from utils.retrieval import VectorStore
from utils.rag_pipeline import generate_answer
import os
import hashlib

# -----------------------------
# CACHE SETUP
# This is to ensure that if you upload the same file, the app re-processing it.
# -----------------------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True) # creating a chache directory

# function to read a file and create a unique hash based on its contents.
def get_file_hash(file):
    file_bytes = file.read()
    file.seek(0) # Moves the file "cursor" back to the start of the file
    return hashlib.md5(file_bytes).hexdigest()

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Personal Research Assitant")
st.title("📄 Personal Research Assitant")

# Smooth scrolling on clicking on links
st.markdown(
    "<style>html { scroll-behavior: smooth; }</style>",
    unsafe_allow_html=True
)

# -----------------------------
# SESSION STATE (a short term memory)
# Since Streamlit re-runs the entire script every time the user clicks a button, a session state helps retain the data
# between clicks
# -----------------------------
# checks whether the vector store exists
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# checks whether the PDFs have been preprocessed
if "processed" not in st.session_state:
    st.session_state.processed = False

# -----------------------------
# COLLAPSABLE SIDEBAR: To upload and process the PDFs
# -----------------------------
st.sidebar.header("Upload Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"], # accept only pdf documents
    accept_multiple_files=True # allow multiple file uploads
)

# Processing only starts when the "Process Documents" button is clicked and there is at least 1 file uploaded
if st.sidebar.button("Process Documents") and uploaded_files:

    all_chunks = [] # stores the text chunks
    all_metadata = [] # stores source information
    cache_keys = [] # stores hashed file data

    with st.spinner("Processing documents..."):

        for file in uploaded_files:
            file_hash = get_file_hash(file) # hash the file
            cache_path = os.path.join(CACHE_DIR, file_hash) 

            # if the file has already been preprocessed
            if os.path.exists(cache_path + ".index"): 
                st.info(f"Loaded from cache: {file.name}")
                store = VectorStore.load(cache_path) # load the existing vector store (the pre-calculated search index)

                # 
                st.session_state.vector_store = store
                st.session_state.processed = True
                continue

            # if the file has NOT already been preprocessed
            text = extract_text_from_pdf(file) # extract the text content from the pdf
            chunks = chunk_text(text) # chunk the text

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({ # for each chunk, store the metadata
                    "source": file.name, # the name of the file the chunk belongs to
                    "chunk_id": i # chunk id
                })

            # Add the processed file info to the cache list
            cache_keys.append((file_hash, file.name, chunks))

        # -------------------------
        # EMBEDDINGS + FAISS BUILD
        # -------------------------
        if all_chunks:
            # convert text chunks to embeddings (numerical vectors)
            embeddings = get_embeddings(all_chunks) 

            dim = len(embeddings[0])
            store = VectorStore(dim) # initializing the vector store (to store the embeddings)
            store.add(embeddings, all_chunks, all_metadata) # adding the embeddings, original text and metadata to the store

            # Saving each processed document's data to the cache folder
            start = 0
            for file_hash, name, chunks in cache_keys:
                count = len(chunks)

                sub_store = VectorStore(dim)
                sub_store.add(
                    embeddings[start:start+count],
                    all_chunks[start:start+count],
                    all_metadata[start:start+count]
                )

                sub_store.save(os.path.join(CACHE_DIR, file_hash))
                start += count

            st.session_state.vector_store = store
            st.session_state.processed = True

    st.success("Documents processed successfully!")

# -----------------------------
# MAIN UI
# -----------------------------
query = st.text_input("Ask a question about the papers:")

# once we have recieved the user questions and the uploaded documents have been processed
if query and st.session_state.processed:

    with st.spinner("Retrieving answer..."):

        # -------------------------
        # RETRIEVAL
        # -------------------------
        query_emb = embed_query(query) # creates an embedding out of the user question
        # Finds the top 4 most relevant chunks from the documents that match the question.
        results = st.session_state.vector_store.search(query_emb, k=4)

        # -------------------------
        # CONSISTENT SOURCE MAPPING : Creates a unique list of which files the relevant chunks came from.
        # -------------------------
        sources = list(dict.fromkeys([r["metadata"]["source"] for r in results]))
        source_map = {source: i + 1 for i, source in enumerate(sources)}

        # Attach ref_id
        for r in results:
            r["ref_id"] = source_map[r["metadata"]["source"]]

        # Sends the relevant chunks and the question to the LLM to generate and write a response.
        answer = generate_answer(results, query)


    st.markdown("<div id='top'></div>", unsafe_allow_html=True)


    st.subheader("Answer")
    st.write(answer)

    # Clickable links that direct to the respective chunks
    st.subheader("Sources")

    for source, ref_id in source_map.items():
        st.markdown(
            f"[Source {ref_id}](#ref_{ref_id}) — {source}"
        )

    # -----------------------------
    # RETRIEVED CHUNKS
    # -----------------------------
    st.subheader("Retrieved Chunks")

    grouped = {}
    for r in results:
        grouped.setdefault(r["ref_id"], []).append(r)

    for ref_id, chunks in grouped.items():

        # Anchor target (end point of the links)
        st.markdown(f"<div id='ref_{ref_id}'></div>", unsafe_allow_html=True)

        with st.expander(f"(Source {ref_id}) Show chunks"): # drop down to reveal the text chunks

            for c in chunks:
                st.markdown(f"**Source file:** {c['metadata']['source']}")
                st.write(c["text"])
                st.write("---")



    st.markdown("[⬆️ Back to Answer](#top)")