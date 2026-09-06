from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


class LLMInterface:
    def __init__(
        self,
        provider="openrouter",
        model_name="deepseek/deepseek-chat",
        temperature=0,
        api_key=None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

        if provider == "openrouter":
            resolved_api_key = api_key or (
                st.secrets.get("OPENROUTER_API_KEY") if st is not None and hasattr(st, "secrets") else None
            ) or os.getenv("OPENROUTER_API_KEY")

            if not resolved_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY introuvable. Ajoutez-la dans votre .env ou dans Streamlit secrets."
                )

            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=resolved_api_key,
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
            )
        else:
            raise ValueError(f"Provider non supporté : {provider}")

    def get_llm(self):
        return self.llm