"""
src/retriever.py — Retriever (Vector Similarity Search with Session-Level Multi-Filter)

Module: Module 7 (Retriever)

Responsibility:
    - Search the ChromaDB vector database for chunks most relevant to a user query
    - Enforce strict session-level document isolation:
      Requires exact matches for BOTH username AND session_id (UUID)
    - Expose standard LangChain retriever interface (as_retriever)
    - Provide retrieve_documents() function returning top-k relevant Document chunks

Imported by:
    - src/rag_pipeline.py (in Module 10)
"""

from typing import List, Optional, Dict, Any
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_core.vectorstores import VectorStoreRetriever
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma

from src.vector_store import load_vector_store


def get_retriever(
    vector_store: Optional[Chroma] = None,
    k: int = 3,
    search_type: str = "similarity",
    username: Optional[str] = None,
    session_id: Optional[str] = None,
) -> VectorStoreRetriever:
    """
    Construct a LangChain VectorStoreRetriever from a ChromaDB vector store instance.
    Applies strict metadata filtering for exact match on username and session_id (UUID).

    Args:
        vector_store (Chroma, optional): Active Chroma vector store instance.
                                         Defaults to loading the persistent store from disk.
        k (int): Number of top relevant document chunks to retrieve (default: 3).
        search_type (str): Retrieval algorithm ('similarity' or 'mmr'). Default: 'similarity'.
        username (str, optional): Target user's username for metadata-filtered retrieval.
        session_id (str, optional): Target chat session UUID for session-level isolation.

    Returns:
        VectorStoreRetriever: Configured LangChain retriever runnable with strict filter.
    """
    store = vector_store or load_vector_store()

    clean_user = username.strip() if username else None
    clean_session_id = session_id.strip() if session_id else None

    search_kwargs: Dict[str, Any] = {"k": k}

    # Strict multi-attribute metadata filter
    if clean_user and clean_session_id:
        search_kwargs["filter"] = {
            "$and": [
                {"username": {"$eq": clean_user}},
                {"session_id": {"$eq": clean_session_id}},
            ]
        }
    elif clean_user:
        search_kwargs["filter"] = {"username": {"$eq": clean_user}}
    elif clean_session_id:
        search_kwargs["filter"] = {"session_id": {"$eq": clean_session_id}}

    retriever = store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return retriever


def retrieve_documents(
    query: str,
    vector_store: Optional[Chroma] = None,
    k: int = 3,
    username: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the top-k most relevant Document chunks for a user search query,
    strictly restricted to the specified username and session_id (UUID).

    Args:
        query (str): The search query or question.
        vector_store (Chroma, optional): Active Chroma vector store instance.
        k (int): Number of top relevant document chunks to return.
        username (str, optional): The logged-in user's username.
        session_id (str, optional): The active chat session UUID.

    Returns:
        List[Document]: List of most relevant LangChain Document chunks belonging to the session.

    Raises:
        ValueError: If query is empty or whitespace only.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    retriever = get_retriever(
        vector_store=vector_store,
        k=k,
        username=username,
        session_id=session_id,
    )
    results = retriever.invoke(query.strip())
    return results
