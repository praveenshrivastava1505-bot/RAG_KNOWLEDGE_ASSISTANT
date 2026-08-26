"""
src/rag_pipeline.py — Complete RAG Pipeline (Orchestrator with Session-Level Data Isolation)

Modules:
    - Module 6: Document Ingestion Pipeline (ingest_document with username + session_id metadata, batching, deduplication)
    - Module 10: Query & Generation Pipeline (ask_question with strict session-level filtering)

Responsibility:
    - Central orchestrator connecting all individual modules together
    - Ingestion Pipeline: Injects both username AND session_id into chunk metadata, splits, embeds, stores in ChromaDB
    - Query Pipeline: Filters retrieval strictly by BOTH username AND session_id, formats prompt, generates grounded answer

Imported by:
    - app.py (Streamlit UI)
    - ingest.py (CLI tool)
    - tests/ (Test suites)
"""

import os
import time
from typing import Dict, Any, List, Optional
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.document_loader import load_document
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, load_vector_store, add_documents
from src.retriever import get_retriever, retrieve_documents
from src.prompt_template import get_rag_prompt_template, format_documents
from src.generator import get_llm


# ============================================================================
# 1. INGESTION PIPELINE (Module 6 with Username & Session-Level Tagging)
# ============================================================================

def ingest_document(
    file_path: str,
    username: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Execute the complete document ingestion pipeline for a single file:
    1. Load text content from file (PDF, DOCX, TXT)
    2. Split text into structured, overlapping chunks
    3. Tag each chunk with BOTH username AND session_id (UUID) for strict session isolation
    4. Generate deterministic unique IDs to prevent duplicates on re-ingestion
    5. Generate vector embeddings in batches of 50 chunks
    6. Save/overwrite chunks in ChromaDB with deterministic IDs and session metadata

    Args:
        file_path (str): Path to the target document.
        username (str, optional): The owner/uploader username for data isolation.
        session_id (str, optional): The target chat session UUID for session isolation.
        chunk_size (int, optional): Custom character chunk size.
        chunk_overlap (int, optional): Custom character chunk overlap.
        batch_size (int): Number of chunks to process per API batch (default: 50).

    Returns:
        Dict[str, Any]: Summary dictionary with ingestion statistics.

    Raises:
        FileNotFoundError: If the document path does not exist.
        ValueError: If document loading or splitting produces no content.
    """
    # Step 1: Document Loading (Module 2)
    raw_docs = load_document(file_path)
    if not raw_docs:
        raise ValueError(f"No text content could be extracted from: '{file_path}'")

    # Step 2: Text Splitting (Module 3)
    chunks = split_documents(
        raw_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"Failed to create chunks from document: '{file_path}'")

    # Step 3: Inject user and session metadata for strict session-level isolation
    clean_user = username.strip() if username else None
    clean_session_id = session_id.strip() if session_id else None

    for chunk in chunks:
        if clean_user:
            chunk.metadata["username"] = clean_user
        if clean_session_id:
            chunk.metadata["session_id"] = clean_session_id

    # Step 4: Generate deterministic unique IDs for all chunks
    total_chunks = len(chunks)
    id_parts = [p for p in (clean_user, clean_session_id, file_path) if p]
    prefix = "_".join(id_parts) if id_parts else file_path
    ids = [f"{prefix}_{i}" for i in range(total_chunks)]

    # Step 5: Embedding Model (Module 4) & Vector Store (Module 5)
    embedding_model = get_embedding_model()

    vector_store = load_vector_store(
        embedding_model=embedding_model,
        persist_directory=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # Process document chunks in batches with deterministic IDs and session metadata
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        vector_store.add_documents(documents=batch, ids=batch_ids)

        processed = min(i + batch_size, total_chunks)

        # If more chunks remain to be processed, wait 60 seconds to reset Gemini API RPM quota
        if processed < total_chunks:
            print(
                f"Rate limit protection: sleeping for 60s... Processed {processed}/{total_chunks} chunks"
            )
            time.sleep(60)

    total_chunks_in_db = vector_store._collection.count()

    return {
        "status": "success",
        "file_path": file_path,
        "username": clean_user,
        "session_id": clean_session_id,
        "raw_pages_loaded": len(raw_docs),
        "chunks_created": len(chunks),
        "total_chunks_in_db": total_chunks_in_db,
    }


# ============================================================================
# 2. QUERY & GENERATION PIPELINE (Module 10 with Session-Level Metadata Filtering)
# ============================================================================

def ask_question(
    question: str,
    k: int = 3,
    vector_store: Optional[Chroma] = None,
    username: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the complete RAG query pipeline with strict session-level data isolation:
    1. Retrieve the top-k most relevant document chunks matching BOTH username AND session_id (Module 7)
    2. Format the chunks into a unified context string (Module 8)
    3. Construct the grounded prompt and send to Gemini LLM (Module 9)
    4. Parse and return the answer string alongside source citations

    Args:
        question (str): The user's query or question.
        k (int): Number of relevant document chunks to retrieve (default: 3).
        vector_store (Chroma, optional): Active vector store instance.
        username (str, optional): Logged-in username to enforce user isolation.
        session_id (str, optional): Active chat session UUID to enforce session isolation.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'question' (str): Original query.
            - 'answer' (str): Generated natural language answer.
            - 'source_documents' (List[Document]): Retrieved chunk citations.
            - 'username' (str): Query username.
            - 'session_id' (str): Query session ID.

    Raises:
        ValueError: If question is empty or ChromaDB contains no data.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    clean_question = question.strip()
    clean_user = username.strip() if username else None
    clean_session_id = session_id.strip() if session_id else None

    # Step 1: Session-Filtered Similarity Search (Module 7: Retriever)
    retrieved_docs: List[Document] = retrieve_documents(
        query=clean_question,
        vector_store=vector_store,
        k=k,
        username=clean_user,
        session_id=clean_session_id,
    )

    # Step 2: Context Formatting (Module 8: Prompt Template)
    context_str = format_documents(retrieved_docs)

    # Step 3: Components for the Chain (Module 8 Prompt + Module 9 LLM + Output Parser)
    prompt_template = get_rag_prompt_template()
    llm = get_llm()
    output_parser = StrOutputParser()

    # Step 4: LCEL Chain Composition: Prompt -> Gemini LLM -> String Output Parser
    rag_chain = prompt_template | llm | output_parser

    # Step 5: Invoke Chain
    answer = rag_chain.invoke(
        {
            "context": context_str,
            "question": clean_question,
        }
    )

    return {
        "question": clean_question,
        "answer": answer.strip(),
        "source_documents": retrieved_docs,
        "username": clean_user,
        "session_id": clean_session_id,
    }
