import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from utils.embeddings import get_embeddings, embed_query
from utils.retrieval import VectorStore
from utils.rag_pipeline import generate_answer

st.set_page_config(page_title="Research Copilot Lite")

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

        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": file.name,
                        "chunk_id": i
                    })

            embeddings = get_embeddings(all_chunks)

            dim = len(embeddings[0])
            store = VectorStore(dim)
            store.add(embeddings, all_chunks, all_metadata)

            st.session_state.vector_store = store
            st.session_state.processed = True

        st.success("Documents processed!")

# Main UI
query = st.text_input("Ask a question about the papers:")

if query and st.session_state.processed:
    with st.spinner("Retrieving answer..."):
        query_emb = embed_query(query)
        results = st.session_state.vector_store.search(query_emb, k=4)

        answer = generate_answer(results, query)

    st.subheader("Answer")
    st.write(answer)

    # Sources
    st.subheader("Sources")
    sources = set([r["metadata"]["source"] for r in results])
    for s in sources:
        st.write(f"- {s}")

# Debug mode (bonus)
if st.checkbox("Show retrieved chunks"):
    if query and st.session_state.processed:
        for r in results:
            st.write("----")
            st.write(r["text"])
            st.write(r["metadata"])