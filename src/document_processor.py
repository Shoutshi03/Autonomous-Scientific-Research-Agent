import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process_pdf(self, file_path, doc_name=None):
        """Charge un PDF et ajoute metadata document."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 🔥 ajouter metadata important
        for doc in documents:
            doc.metadata["source_doc"] = doc_name or os.path.basename(file_path)

        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_multiple_pdfs(self, file_paths):
        all_chunks = []

        for path in file_paths:
            doc_name = os.path.basename(path)
            chunks = self.process_pdf(path, doc_name=doc_name)
            all_chunks.extend(chunks)

        return all_chunks