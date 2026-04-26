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
# -----------------------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file):
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Personal Research Copilot")
st.title("📄 Research Copilot Lite")

# Smooth scrolling
st.markdown(
    "<style>html { scroll-behavior: smooth; }</style>",
    unsafe_allow_html=True
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed" not in st.session_state:
    st.session_state.processed = False

# -----------------------------
# SIDEBAR: UPLOAD + PROCESS
# -----------------------------
st.sidebar.header("Upload Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents") and uploaded_files:

    all_chunks = []
    all_metadata = []
    cache_keys = []

    with st.spinner("Processing documents..."):

        for file in uploaded_files:
            file_hash = get_file_hash(file)
            cache_path = os.path.join(CACHE_DIR, file_hash)

            # -------------------------
            # LOAD FROM CACHE
            # -------------------------
            if os.path.exists(cache_path + ".index"):
                st.info(f"Loaded from cache: {file.name}")
                store = VectorStore.load(cache_path)

                st.session_state.vector_store = store
                st.session_state.processed = True
                continue

            # -------------------------
            # PROCESS PDF
            # -------------------------
            text = extract_text_from_pdf(file)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": file.name,
                    "chunk_id": i
                })

            cache_keys.append((file_hash, file.name, chunks))

        # -------------------------
        # EMBEDDINGS + FAISS BUILD
        # -------------------------
        if all_chunks:
            embeddings = get_embeddings(all_chunks)

            dim = len(embeddings[0])
            store = VectorStore(dim)
            store.add(embeddings, all_chunks, all_metadata)

            # Save per-document cache
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

if query and st.session_state.processed:

    with st.spinner("Retrieving answer..."):

        # -------------------------
        # RETRIEVAL
        # -------------------------
        query_emb = embed_query(query)
        results = st.session_state.vector_store.search(query_emb, k=4)

        # -------------------------
        # CONSISTENT SOURCE MAPPING
        # -------------------------
        sources = list(dict.fromkeys([r["metadata"]["source"] for r in results]))
        source_map = {source: i + 1 for i, source in enumerate(sources)}

        # Attach ref_id
        for r in results:
            r["ref_id"] = source_map[r["metadata"]["source"]]

        # -------------------------
        # RAG ANSWER
        # -------------------------
        answer = generate_answer(results, query)

    # -----------------------------
    # ANCHOR FOR BACK BUTTON
    # -----------------------------
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)

    # -----------------------------
    # ANSWER SECTION
    # -----------------------------
    st.subheader("Answer")
    st.write(answer)

    # -----------------------------
    # SOURCES (CLICKABLE)
    # -----------------------------
    st.subheader("Sources")

    for source, ref_id in source_map.items():
        st.markdown(
            f"[Source {ref_id}](#ref_{ref_id}) — {source}"
        )

    # -----------------------------
    # RETRIEVED CHUNKS (FIXED UX)
    # -----------------------------
    st.subheader("Retrieved Chunks")

    grouped = {}
    for r in results:
        grouped.setdefault(r["ref_id"], []).append(r)

    for ref_id, chunks in grouped.items():

        # Anchor target
        st.markdown(f"<div id='ref_{ref_id}'></div>", unsafe_allow_html=True)

        with st.expander(f"(Source {ref_id}) Show chunks"):

            for c in chunks:
                st.markdown(f"**Source file:** {c['metadata']['source']}")
                st.write(c["text"])
                st.write("---")

    # -----------------------------
    # BACK TO TOP
    # -----------------------------
    st.markdown("[⬆️ Back to Answer](#top)")