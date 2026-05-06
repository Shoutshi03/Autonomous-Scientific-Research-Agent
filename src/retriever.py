from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retrieval = FAISS + BM25 + CrossEncoder reranking
    Compatible LangChain 0.2.x
    """

    vector_manager: Any
    k: int = 5

    # champs Pydantic obligatoires
    faiss_retriever: Optional[Any] = None
    bm25: Optional[Any] = None
    reranker: Optional[Any] = None
    initialized: bool = False

    def _init_components(self):
        """Initialisation lazy (une seule fois)"""

        if self.initialized:
            return

        # FAISS dense retrieval
        self.faiss_retriever = self.vector_manager.get_retriever(
            search_kwargs={"k": self.k}
        )

        # BM25 sparse retrieval
        docs = list(self.vector_manager.get_all_documents())

        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = self.k

        # Cross Encoder reranker
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.initialized = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ) -> List[Document]:

        self._init_components()

        # retrieval FAISS
        docs_faiss = self.faiss_retriever.invoke(query)

        # retrieval BM25
        docs_bm25 = self.bm25.invoke(query)

        # fusion
        docs = docs_faiss + docs_bm25

        # remove duplicates
        unique_docs = list(
            {doc.page_content: doc for doc in docs}.values()
        )

        # reranking
        pairs = [(query, doc.page_content) for doc in unique_docs]
        scores = self.reranker.predict(pairs)

        ranked_docs = sorted(
            zip(unique_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked_docs[:self.k]]