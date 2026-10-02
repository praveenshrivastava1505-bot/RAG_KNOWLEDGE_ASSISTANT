"""
src/generator.py — LLM Generator (Google Gemini API via LangChain)

Module: Module 9 (LLM Generator)

Responsibility:
    - Initialize and configure the Google Gemini LLM (ChatGoogleGenerativeAI)
    - Default to LLM_MODEL ("gemini-1.5-flash") and LLM_TEMPERATURE (0.2) from src/config.py
    - Expose get_llm() factory function for LCEL pipelines (prompt | llm | StrOutputParser)
    - Provide generate_response() helper to generate textual answers

What is the Generator?
    The Generator is the "brain/speaker" of the RAG assistant:
    1. It receives the grounded prompt prepared by Module 8 (containing retrieved context + question).
    2. Sends the prompt to Google Gemini 1.5 Flash via ChatGoogleGenerativeAI.
    3. Returns a well-formatted, factual natural language answer to the user.

Imported by:
    - src/rag_pipeline.py (in Module 10)
"""

import os
from typing import Optional, List
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    MODELS_TO_TRY,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)

load_dotenv()

# Candidate models list with fallback options to prevent 404 errors
models_to_try: List[str] = ["gemini-flash-latest", "gemini-3.8-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    candidate_models: Optional[List[str]] = None,
) -> ChatGoogleGenerativeAI:
    """
    Initialize and return a Google Gemini Chat LLM instance with fallback support.

    Args:
        model_name (str, optional): Gemini model name (default: "gemini-flash-latest").
        temperature (float, optional): Sampling temperature (default: 0.2).
        api_key (str, optional): Google API Key.
        max_output_tokens (int, optional): Max output tokens (default: 8192).
        candidate_models (List[str], optional): List of models to try in order.

    Returns:
        ChatGoogleGenerativeAI: Configured LangChain Chat model instance.

    Raises:
        ValueError: If no valid Google API key is found or no model could be initialized.
    """
    key = (
        api_key
        or GOOGLE_API_KEY
        or os.getenv("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        try:
            import streamlit as st
            if "GOOGLE_API_KEY" in st.secrets:
                key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass

    if not key:
        raise ValueError("Could not initialize any Gemini model. Check your API key.")

    temp = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_output_tokens or LLM_MAX_OUTPUT_TOKENS
    candidates = [model_name] if model_name else (candidate_models or models_to_try)

    llm = None
    for m in candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=m,
                google_api_key=key,
                temperature=temp,
                max_tokens=max_tokens,
                max_output_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(f"Model {m} failed: {e}")
            continue

    if not llm:
        raise ValueError("Could not initialize any Gemini model. Check your API key.")

    return llm


def generate_response(
    prompt: str,
    llm: Optional[ChatGoogleGenerativeAI] = None,
    candidate_models: Optional[List[str]] = None,
) -> str:
    """
    Send a direct text prompt to the Gemini LLM with automatic fallback on 404 errors.

    Args:
        prompt (str): Text prompt or formatted message.
        llm (ChatGoogleGenerativeAI, optional): Active LLM instance.
        candidate_models (List[str], optional): Fallback models if invocation fails.

    Returns:
        str: Clean string content of the model's response.
    """
    candidates = candidate_models or models_to_try
    for m in candidates:
        try:
            model = llm if (llm and m == candidates[0]) else get_llm(model_name=m)
            response = model.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content).strip()
            return str(response).strip()
        except Exception as e:
            print(f"Model {m} invocation failed: {e}. Trying fallback...")
            continue

    raise RuntimeError("All Gemini models in fallback list failed to generate a response.")

