import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from src.config import settings
from src.document_processor import DocumentProcessor
from src.llm_interface import LLMInterface
from src.rag_chain import RAGChainManager
from src.retriever import HybridRetriever
from src.vector_store import VectorStoreManager

load_dotenv()

st.set_page_config(page_title=settings.app_title, layout="wide")


@st.cache_resource
def init_app():
    doc_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_file_size_mb=settings.max_file_size_mb,
    )
    vector_manager = VectorStoreManager(
        persist_directory=settings.vector_store_dir_str,
        allow_dangerous_deserialization=settings.allow_dangerous_deserialization,
    )
    llm_interface = LLMInterface(
        provider=settings.llm_provider,
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    return doc_processor, vector_manager, llm_interface


try:
    doc_processor, vector_manager, llm_interface = init_app()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


def stream_response(response_stream, placeholder):
    full = ""
    for chunk in response_stream:
        if isinstance(chunk, dict) and "answer" in chunk:
            token = chunk["answer"]
            full += token
            placeholder.markdown(full + "▌")
            time.sleep(0.02)
    return full


st.title(settings.app_title)

with st.sidebar:
    st.header("Documents")
    st.caption(f"Index : {settings.vector_store_dir_str}")
    uploaded_files = st.file_uploader("Téléverser des PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Indexer les documents") and uploaded_files:
        all_chunks = []
        doc_names = []

        with st.spinner("Extraction et indexation des PDF..."):
            try:
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        temp_path = tmp.name

                    chunks = doc_processor.process_pdf(temp_path, doc_name=file.name)
                    for chunk in chunks:
                        chunk.metadata["source"] = file.name
                    all_chunks.extend(chunks)
                    doc_names.append(file.name)
                    os.unlink(temp_path)

                if not all_chunks:
                    raise ValueError("Aucun contenu exploitable n'a été trouvé dans les fichiers PDF fournis.")

                vector_manager.create_vector_store(all_chunks)
                vector_manager.save_vector_store()
                st.session_state.vector_ready = True
                st.session_state.doc_names = doc_names
                st.success(f"{len(all_chunks)} chunks indexés depuis {len(doc_names)} document(s).")
            except Exception as exc:
                st.error(f"Erreur lors de l'indexation : {exc}")

    if st.button("Vider l'index"):
        st.session_state.vector_ready = False
        st.session_state.doc_names = []
        try:
            if hasattr(vector_manager, "vector_store") and vector_manager.vector_store is not None:
                vector_manager.vector_store = None
            st.success("Index local réinitialisé.")
        except Exception as exc:
            st.error(f"Impossible de réinitialiser l'index : {exc}")


if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = bool(vector_manager.load_vector_store())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question scientifique..."):
    if not st.session_state.vector_ready:
        st.warning("Indexez d'abord au moins un document PDF avant de poser une question.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                vector_manager.load_vector_store()
                hybrid = HybridRetriever(vector_manager=vector_manager, k=settings.max_retrieved_docs)
                llm = llm_interface.get_llm()
                rag = RAGChainManager(llm, hybrid)
                chain = rag.create_chain()

                response_stream = chain.stream({"input": prompt})
                answer = stream_response(response_stream, placeholder)
                placeholder.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })
            except Exception as exc:
                st.error(f"Erreur de génération : {exc}")