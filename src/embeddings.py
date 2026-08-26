"""
src/embeddings.py — Local Embedding Generation (HuggingFace)

Module: Module 4 (Embedding Generation)

Responsibility:
    - Initialize and configure local HuggingFace text embedding model
    - Use HuggingFaceEmbeddings from langchain-huggingface
    - Standardize embedding model configuration (all-MiniLM-L6-v2)
    - Runs 100% locally with zero API rate limits or daily quotas
    - Provide centralized get_embeddings() and get_embedding_model() functions

What is an Embedding?
    An embedding converts text into a high-dimensional vector (a list of numbers,
    e.g. [0.024, -0.015, 0.089, ...]).
    Semantically similar texts produce vectors that are close together in geometric space,
    enabling mathematical similarity search.

Imported by:
    - src/vector_store.py (in Module 5)
    - src/rag_pipeline.py (in Module 6 / Module 10)
    - src/retriever.py (in Module 7)
"""

from typing import Optional
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
)


def get_embeddings(
    model_name: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """
    Initialize and return a local HuggingFace embedding model instance.

    Args:
        model_name (str, optional): The HuggingFace embedding model name or path.
                                     Defaults to EMBEDDING_MODEL from config.py ('all-MiniLM-L6-v2').

    Returns:
        HuggingFaceEmbeddings: Configured LangChain HuggingFace embedding model instance.
    """
    model = model_name or EMBEDDING_MODEL
    embeddings = HuggingFaceEmbeddings(model_name=model)
    return embeddings


def get_embedding_model(
    model_name: Optional[str] = None,
    **kwargs,
) -> HuggingFaceEmbeddings:
    """
    Alias for get_embeddings() to ensure seamless backward compatibility with existing modules.

    Args:
        model_name (str, optional): The HuggingFace embedding model name.

    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance.
    """
    return get_embeddings(model_name=model_name)
