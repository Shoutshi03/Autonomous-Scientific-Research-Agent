from __future__ import annotations

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="data/faiss_index",
        allow_dangerous_deserialization=False,
    ):
        self.model_name = model_name
        self.persist_directory = str(Path(persist_directory))
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vector_store = None
        self._ensure_directory()

    def _ensure_directory(self):
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    def create_vector_store(self, chunks):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def save_vector_store(self):
        if self.vector_store:
            self.vector_store.save_local(self.persist_directory)

    def load_vector_store(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=self.allow_dangerous_deserialization,
            )
            return self.vector_store
        return None

    def add_documents(self, chunks):
        if self.vector_store:
            self.vector_store.add_documents(chunks)
        else:
            self.create_vector_store(chunks)

        self.save_vector_store()

    def get_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        if not self.vector_store:
            raise ValueError("Vector store non initialisé.")

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_all_documents(self):
        if not self.vector_store:
            return []

        return list(self.vector_store.docstore._dict.values())

    def is_ready(self):
        return self.vector_store is not None or os.path.exists(self.persist_directory)

    