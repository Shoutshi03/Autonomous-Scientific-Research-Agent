# Scientific Research RAG Agent

A production-oriented Streamlit application for scientific PDF analysis. It combines hybrid retrieval (FAISS + BM25), optional reranking, and a grounded generation pipeline to answer research questions from a document corpus.

## Features

- Upload multiple PDF files
- Extract text and chunk long scientific documents
- Index documents in a FAISS vector store
- Combine dense + lexical retrieval
- Run grounded question answering with OpenRouter models
- Stream responses in the UI in real time
- Configure key parameters using environment variables

## Architecture

- `app.py`: Streamlit interface
- `src/document_processor.py`: PDF ingestion and chunking
- `src/vector_store.py`: FAISS persistence and retrieval setup
- `src/retriever.py`: hybrid retrieval with BM25 + CrossEncoder reranking
- `src/llm_interface.py`: LLM provider abstraction
- `src/rag_chain.py`: RAG orchestration and prompt template
- `src/config.py`: environment-based configuration management

## Prerequisites

- Python 3.10+
- pip
- A valid OpenRouter API key

## Installation

```bash
git clone https://github.com/Shoutshi03/Autonomous-Scientific-Research-Agent.git
cd Autonomous-Scientific-Research-Agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

```bash
copy .env.example .env
```

Example:

```env
OPENROUTER_API_KEY=your_api_key_here
LLM_MODEL=deepseek/deepseek-chat
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVED_DOCS=5
```

## Run locally

```bash
streamlit run app.py
```

## Production notes

- Keep API keys in environment variables or your deployment secret manager
- Store the FAISS index in a persistent volume if deployed in a container
- Validate uploaded files server-side before indexing
- Monitor model token usage and retrieval quality in production
