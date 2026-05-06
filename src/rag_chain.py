from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGChainManager:
    def __init__(self, llm, retriever, reranker=None):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker

        # PROMPT OPTIMISÉ :
        self.system_prompt = """
            RÔLE :
            Vous êtes un assistant expert en recherche scientifique spécialisé dans l’analyse d’articles académiques et la synthèse d’informations complexes.

            OBJECTIF :
            Produire des réponses rigoureuses, précises et fidèles aux documents fournis.

            CONTEXTE :
            Les informations proviennent exclusivement d’articles scientifiques (PDF) fournis via un système RAG.

            RÈGLES STRICTES :
            - Utilisez UNIQUEMENT les informations du contexte.
            - Interdiction totale d’inventer ou compléter avec des connaissances externes.
            - Si l’information est absente :
            → Répondez EXACTEMENT :
            "Je ne sais pas sur la base des documents fournis."
            - Priorité à la précision plutôt qu’à la complétude.

            MÉTHODOLOGIE INTERNE (NE PAS AFFICHER) :
            1. Comprendre la question
            2. Identifier les passages pertinents
            3. Croiser les informations
            4. Synthétiser

            STYLE :
            - Académique, clair, structuré
            - Pas de blabla inutile
            - Pas de généralités vagues

            FORMAT DE RÉPONSE (OBLIGATOIRE) :

            ### Réponse directe
            Réponse concise et précise à la question.

            ### Explication détaillée
            Analyse approfondie basée sur le contexte.

            ### Éléments clés extraits
            - Point 1
            - Point 2
            - Point 3
            
            CONTEXTE DOCUMENTAIRE :
            {context}
            """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])

    def create_chain(self):
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        return create_retrieval_chain(self.retriever, doc_chain)

    def get_response(self, chain, query):
        return chain.invoke({"input": query})

    def get_reranked_docs(self, query, docs):
        if self.reranker:
            return self.reranker.rerank(query, docs)
        return docs