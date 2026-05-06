import streamlit as st
import tempfile
import os
import time
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_interface import LLMInterface
from src.rag_chain import RAGChainManager
from src.retriever import HybridRetriever

st.set_page_config(page_title="Scientific Research RAG Agent", layout="wide")

load_dotenv()

# Init
@st.cache_resource
def init():
    return DocumentProcessor(), VectorStoreManager(), LLMInterface()

doc_processor, vector_manager, llm_interface = init()


# 🔥 Streaming function
def stream_response(response_stream, placeholder):
    full = ""
    for chunk in response_stream:
        if isinstance(chunk, dict) and "answer" in chunk:
            token = chunk["answer"]
            full += token
            placeholder.markdown(full + "▌")
            time.sleep(0.02)
    return full


st.title("Scientific Research RAG agent")

# Sidebar
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

    if st.button("Traiter") and uploaded_files:
        all_chunks = []
        doc_names = []

        with st.spinner("Traitement..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    path = tmp.name

                chunks = doc_processor.process_pdf(path)

                # 🔥 ajouter metadata doc
                for c in chunks:
                    c.metadata["source"] = file.name

                all_chunks.extend(chunks)
                doc_names.append(file.name)

                os.unlink(path)

        vector_manager.create_vector_store(all_chunks)
        vector_manager.save_vector_store()

        st.session_state.vector_ready = True
        st.session_state.doc_names = doc_names

        st.success("Documents indexés")

# Load existing index
if "vector_ready" not in st.session_state:
    if vector_manager.load_vector_store():
        st.session_state.vector_ready = True
    else:
        st.session_state.vector_ready = False


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
if prompt := st.chat_input("Pose ta question scientifique..."):
    if not st.session_state.vector_ready:
        st.warning("Charge d'abord des documents")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            try:
                vector_manager.load_vector_store()

                # 🔥 HYBRID RETRIEVER
                hybrid = HybridRetriever(vector_manager=vector_manager, k=5)

                llm = llm_interface.get_llm()

                # 🔥 RAG CHAIN
                rag = RAGChainManager(llm, hybrid)
                chain = rag.create_chain()

                # 🔥 STREAMING
                response_stream = chain.stream({"input": prompt})
                answer = stream_response(response_stream, placeholder)

                placeholder.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(str(e))