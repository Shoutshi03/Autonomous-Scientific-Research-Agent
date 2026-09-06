from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, max_file_size_mb=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_mb = max_file_size_mb
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def _validate_pdf(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Le fichier {path.name} n'est pas un PDF valide.")

        if path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
            raise ValueError(
                f"Le fichier {path.name} dépasse {self.max_file_size_mb} Mo. "
                "Réduisez sa taille ou divisez-le en morceaux."
            )

        return path

    def process_pdf(self, file_path, doc_name=None):
        """Charge un PDF, nettoie les métadonnées et découpe les documents en chunks."""
        path = self._validate_pdf(file_path)
        loader = PyPDFLoader(str(path))
        documents = loader.load()

        if not documents:
            raise ValueError(f"Aucun contenu n'a pu être extrait du PDF {path.name}.")

        for doc in documents:
            metadata = doc.metadata or {}
            metadata["source_doc"] = doc_name or path.name
            metadata["source"] = metadata.get("source", doc_name or path.name)
            metadata["file_name"] = path.name
            metadata["page_count"] = len(documents)
            doc.metadata = metadata

        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            raise ValueError(f"Le PDF {path.name} n'a produit aucun chunk exploitable.")

        for chunk in chunks:
            chunk.metadata.setdefault("source_doc", doc_name or path.name)
            chunk.metadata.setdefault("source", doc_name or path.name)
            chunk.metadata.setdefault("file_name", path.name)

        return chunks

    def process_multiple_pdfs(self, file_paths):
        all_chunks = []

        for path in file_paths:
            doc_name = os.path.basename(path)
            chunks = self.process_pdf(path, doc_name=doc_name)
            all_chunks.extend(chunks)

        return all_chunks