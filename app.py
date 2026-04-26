import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from utils.embeddings import get_embeddings, embed_query
from utils.retrieval import VectorStore
from utils.rag_pipeline import generate_answer
import os
import hashlib

# creating a cache directory
# caching based on file content (by hashing the pdf) so that the same pdf is not reprocessed
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# hash a pdf file
def get_file_hash(file):
    file_bytes = file.read()
    file.seek(0)  # reset pointer after reading
    return hashlib.md5(file_bytes).hexdigest()

st.set_page_config(page_title="Personal Research Copilot")

st.title("Research Copilot Lite")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed" not in st.session_state:
    st.session_state.processed = False

# Sidebar
st.sidebar.header("Upload Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if uploaded_files:

        all_chunks = []
        all_metadata = []
        cache_keys = []

        with st.spinner("Processing documents..."):

            for file in uploaded_files:
                file_hash = get_file_hash(file)
                cache_path = os.path.join(CACHE_DIR, file_hash)

                if os.path.exists(cache_path + ".index"):
                    st.info(f"Loaded from cache: {file.name}")
                    store = VectorStore.load(cache_path)

                    st.session_state.vector_store = store
                    st.session_state.processed = True
                    continue

                # Otherwise process normally
                text = extract_text_from_pdf(file)
                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": file.name,
                        "chunk_id": i
                    })

                cache_keys.append((file_hash, file.name, chunks))

            # If new docs exist, then embed them
            if all_chunks:
                embeddings = get_embeddings(all_chunks)

                dim = len(embeddings[0])
                store = VectorStore(dim)
                store.add(embeddings, all_chunks, all_metadata)

                # Save cache per file
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

        st.success("Documents processed!")




# Main UI
query = st.text_input("Ask a question about the papers:")

if query and st.session_state.processed:
    with st.spinner("Retrieving answer..."):
        query_emb = embed_query(query)
        results = st.session_state.vector_store.search(query_emb, k=4)

        # answer = generate_answer(results, query)
        answer, source_map = generate_answer(results, query)

    st.markdown("<a id='top'></a>", unsafe_allow_html=True)
    
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    # Reverse mapping: number → filename
    # reverse_map = {v: k for k, v in source_map.items()}

    # for ref_id in sorted(reverse_map.keys()):
    #     st.write(f"(Ref {ref_id}) {reverse_map[ref_id]}")

    reverse_map = {v: k for k, v in source_map.items()}

    for ref_id in sorted(reverse_map.keys()):
        st.markdown(
            f"<a href='#ref_{ref_id}'>(Source {ref_id})</a> {reverse_map[ref_id]}",
            unsafe_allow_html=True
        )




    # # Sources
    # st.subheader("Sources")
    # sources = set([r["metadata"]["source"] for r in results])
    # for s in sources:
    #     st.write(f"- {s}")

# Show chunks from documents
# if st.checkbox("Show retrieved chunks"):
#     if query and st.session_state.processed:
#         for r in results:
#             st.write("----")
#             st.write(r["text"])
#             st.write(r["metadata"])

if st.checkbox("Show retrieved chunks"):
    st.subheader("Retrieved Chunks")

    for i, r in enumerate(results):
        ref_id = list(source_map.values())[i] if i < len(source_map) else i+1

        st.markdown(f"<a id='ref_{ref_id}'></a>", unsafe_allow_html=True)

        st.markdown(f"### (Ref {ref_id}) - {r['metadata']['source']}")
        st.write(r["text"])
        st.write("---")


st.markdown(
    "<a href='#top'>⬆️ Back to Answer</a>",
    unsafe_allow_html=True
)
