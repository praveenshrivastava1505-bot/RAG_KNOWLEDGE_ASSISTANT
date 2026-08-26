"""
src/vector_store.py — Vector Store (ChromaDB)

Module: Module 5 (Vector Store)

Responsibility:
    - Create, persist, and load local ChromaDB vector databases
    - Store document chunks alongside their mathematical embeddings and metadata
    - Use LangChain's Chroma wrapper (langchain-chroma)
    - Default to CHROMA_DB_DIR and CHROMA_COLLECTION_NAME from src/config.py

What is a Vector Database?
    A vector database indexes high-dimensional vectors (from Module 4) so that
    when a user asks a question, it can perform fast mathematical similarity
    search (e.g. Cosine Similarity) to find the most relevant chunks in milliseconds.

Imported by:
    - src/retriever.py (in Module 7)
    - src/rag_pipeline.py (in Module 6 / Module 10)
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.embeddings import get_embedding_model


def create_vector_store(
    documents: List[Document],
    embedding_model: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Create a new ChromaDB vector store from a list of document chunks and persist it to disk.

    Args:
        documents (List[Document]): List of chunked Document objects from Module 3.
        embedding_model (Embeddings, optional): LangChain embedding model instance.
                                               Defaults to get_embedding_model() from Module 4.
        persist_directory (str, optional): Folder path where ChromaDB saves data.
                                           Defaults to CHROMA_DB_DIR from config.py.
        collection_name (str, optional): Name of the Chroma collection.
                                         Defaults to CHROMA_COLLECTION_NAME from config.py.

    Returns:
        Chroma: Initialized and populated Chroma vector store instance.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot create vector store: documents list is empty.")

    embeddings = embedding_model or get_embedding_model()
    directory = persist_directory or CHROMA_DB_DIR
    collection = collection_name or CHROMA_COLLECTION_NAME

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=directory,
        collection_name=collection,
    )

    return vector_store


def load_vector_store(
    embedding_model: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Load an existing persistent ChromaDB vector store from disk.

    Args:
        embedding_model (Embeddings, optional): LangChain embedding model instance.
                                               Defaults to get_embedding_model() from Module 4.
        persist_directory (str, optional): Folder path where ChromaDB is saved.
                                           Defaults to CHROMA_DB_DIR from config.py.
        collection_name (str, optional): Name of the Chroma collection.
                                         Defaults to CHROMA_COLLECTION_NAME from config.py.

    Returns:
        Chroma: Loaded Chroma vector store instance ready for querying.
    """
    embeddings = embedding_model or get_embedding_model()
    directory = persist_directory or CHROMA_DB_DIR
    collection = collection_name or CHROMA_COLLECTION_NAME

    vector_store = Chroma(
        persist_directory=directory,
        embedding_function=embeddings,
        collection_name=collection,
    )

    return vector_store


def add_documents(
    vector_store: Chroma,
    documents: List[Document],
) -> List[str]:
    """
    Add additional document chunks to an existing ChromaDB vector store.

    Args:
        vector_store (Chroma): The active Chroma vector store instance.
        documents (List[Document]): New document chunks to embed and store.

    Returns:
        List[str]: List of unique IDs assigned to the added documents.
    """
    if not documents:
        return []

    return vector_store.add_documents(documents=documents)
