from langchain_openai import ChatOpenAI
import os
import streamlit as st


class LLMInterface:
    def __init__(
        self,
        provider="openrouter",
        model_name="deepseek/deepseek-chat",
        temperature=0
    ):

        if provider == "openrouter":

            # priorité aux secrets Streamlit
            api_key = st.secrets.get(
                "OPENROUTER_API_KEY",
                os.getenv("OPENROUTER_API_KEY")
            )

            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY introuvable."
                )

            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                streaming=True
            )

        else:
            raise ValueError(f"Provider non supporté : {provider}")

    def get_llm(self):
        return self.llm