"""
src/generator.py — LLM Generator (Google Gemini)

Module: Module 9 (LLM Generator)

Responsibility:
    - Initialize and configure the Google Gemini chat model (ChatGoogleGenerativeAI)
    - Default to LLM_MODEL (gemini-2.5-flash) and LLM_TEMPERATURE from src/config.py
    - Expose get_llm() factory function for the LCEL pipeline
    - Provide generate_response() helper to generate textual answers

What is the Generator?
    The Generator is the "brain/speaker" of the RAG assistant:
    1. It receives the grounded prompt prepared by Module 8 (containing retrieved context + question).
    2. Sends the prompt to Google Gemini 2.5 Flash via the Gemini Developer API.
    3. Returns a well-formatted, factual natural language answer to the user.

Imported by:
    - src/rag_pipeline.py (in Module 10)
"""

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """
    Initialize and return a Google Gemini Chat LLM instance.

    Args:
        model_name (str, optional): Gemini model identifier.
                                     Defaults to LLM_MODEL from config.py ('gemini-2.5-flash').
        temperature (float, optional): Sampling temperature (0.0 to 1.0).
                                       Defaults to LLM_TEMPERATURE from config.py (0.3).
        api_key (str, optional): Google API key.
                                 Defaults to GOOGLE_API_KEY from config.py.

    Returns:
        ChatGoogleGenerativeAI: Configured LangChain Chat model instance.

    Raises:
        ValueError: If no valid Google API key is configured.
    """
    key = api_key or GOOGLE_API_KEY
    if not key:
        raise ValueError(
            "Google API key not found. Please ensure GOOGLE_API_KEY or GEMINI_API_KEY "
            "is set in your .env file."
        )

    model = model_name or LLM_MODEL
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temp,
    )

    return llm


def generate_response(
    prompt: str,
    llm: Optional[ChatGoogleGenerativeAI] = None,
) -> str:
    """
    Send a direct text prompt to the Gemini LLM and return the generated text response.

    Args:
        prompt (str): Text prompt or formatted message.
        llm (ChatGoogleGenerativeAI, optional): Active LLM instance.

    Returns:
        str: Clean string content of the model's response.
    """
    model = llm or get_llm()
    response = model.invoke(prompt)
    return response.content
