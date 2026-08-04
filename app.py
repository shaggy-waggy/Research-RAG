import os
import hashlib
import streamlit as st

from rag import RAGPipeline

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# Streamlit App
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📄"
)


st.title("DocuChat-Question Answering")

st.write(
    "Upload a PDF and ask questions based on its content."
)


def build_document_key(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    return file_hash, f"{uploaded_file.name}:{file_hash}"


def save_uploaded_pdf(uploaded_file, file_hash):
    file_name = f"{file_hash[:16]}_{uploaded_file.name}"
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())

    return file_path


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:
    file_hash, doc_key = build_document_key(
        uploaded_file
    )
    file_path = save_uploaded_pdf(
        uploaded_file,
        file_hash
    )

    if st.session_state.get("active_doc_key") != doc_key:
        with st.spinner(
            "Processing document..."
        ):
            rag = RAGPipeline(
                chunk_size=512,
                embedding_model="all-MiniLM-L6-v2",
                retriever_type="mmr"
            )

            chain = rag.process_pdf(
                file_path
            )

        st.session_state["active_doc_key"] = doc_key
        st.session_state["chain"] = chain
        st.success(
            "Document is ready!"
        )
    else:
        st.info(
            "Using cached document. Ask your questions below."
        )

    chain = st.session_state.get("chain")

    # Question input
    question = st.text_input(
        "Ask a question about the document:"
    )


    if question:
        with st.spinner(
            "Generating answer..."
        ):

            response = chain.invoke(
                {
                    "input": question
                }
            )

        st.subheader(
            "Answer"
        )
        st.write(
            response["answer"]
        )

        # Show retrieved chunks
        with st.expander(
            "View Retrieved Context"
        ):

            for i, doc in enumerate(
                response["context"]
            ):

                st.write(
                    f"### Chunk {i+1}"
                )

                st.write(
                    doc.page_content
                )