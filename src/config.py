from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_title: str = "Scientific Research RAG Agent"
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    vector_store_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "faiss_index")
    llm_provider: str = "openrouter"
    llm_model: str = "deepseek/deepseek-chat"
    llm_temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieved_docs: int = 5
    max_file_size_mb: int = 20
    debug: bool = False
    allow_dangerous_deserialization: bool = False

    @property
    def vector_store_dir_str(self) -> str:
        return str(self.vector_store_dir)


def get_settings() -> Settings:
    return Settings(
        app_title=os.getenv("APP_TITLE", "Scientific Research RAG Agent"),
        llm_provider=os.getenv("LLM_PROVIDER", "openrouter"),
        llm_model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        max_retrieved_docs=int(os.getenv("MAX_RETRIEVED_DOCS", "5")),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "20")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        allow_dangerous_deserialization=os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "false").lower() == "true",
    )


settings = get_settings()
