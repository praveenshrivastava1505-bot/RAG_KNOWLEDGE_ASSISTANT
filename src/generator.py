"""
src/generator.py — Pure Hugging Face Endpoint LLM Generator

Module: Module 9 (LLM Generator)

Responsibility:
    - Initialize and configure pure HuggingFaceEndpoint from langchain_huggingface
    - Strictly use the official Hugging Face Serverless API with HUGGINGFACEHUB_API_TOKEN
    - Use the stable free-tier model: repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    - Expose get_llm() factory function for LCEL pipelines (prompt | llm | StrOutputParser)
    - Provide generate_response() helper to generate textual answers

Imported by:
    - src/rag_pipeline.py (in Module 10)
"""

from typing import Optional
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEndpoint

from src.config import (
    HUGGINGFACEHUB_API_TOKEN,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> HuggingFaceEndpoint:
    """
    Initialize and return a pure HuggingFaceEndpoint LLM instance.

    Args:
        model_name (str, optional): Hugging Face model repository identifier.
                                     Defaults to LLM_MODEL from config.py ('mistralai/Mistral-7B-Instruct-v0.2').
        temperature (float, optional): Sampling temperature (0.0 to 1.0).
                                       Defaults to LLM_TEMPERATURE from config.py (0.3).
        api_key (str, optional): Hugging Face API token.
                                 Defaults to HUGGINGFACEHUB_API_TOKEN from config.py.
        max_new_tokens (int, optional): Maximum new tokens to generate (default: 1024 from config.py).

    Returns:
        HuggingFaceEndpoint: Configured pure LangChain Hugging Face Endpoint instance.

    Raises:
        ValueError: If no valid Hugging Face API token is configured.
    """
    token = api_key or HUGGINGFACEHUB_API_TOKEN
    if not token:
        raise ValueError(
            "Hugging Face API token not found. Please ensure HUGGINGFACEHUB_API_TOKEN "
            "is set in your .env file."
        )

    model = model_name or LLM_MODEL
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_TOKENS

    llm = HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=token,
        temperature=temp,
        max_new_tokens=tokens,
        timeout=120,
        task="conversational",
    )

    return llm


def generate_response(
    prompt: str,
    llm: Optional[HuggingFaceEndpoint] = None,
) -> str:
    """
    Send a direct text prompt to the Hugging Face LLM and return the generated text response.

    Args:
        prompt (str): Text prompt or formatted message.
        llm (HuggingFaceEndpoint, optional): Active LLM instance.

    Returns:
        str: Clean string content of the model's response.
    """
    model = llm or get_llm()
    response = model.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()
