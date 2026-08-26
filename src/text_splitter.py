"""
src/text_splitter.py — Text Splitting (Chunking)

Module: Module 3 (Text Splitting)

Responsibility:
    - Split raw LangChain Document objects into smaller, overlapping chunks
    - Use LangChain's RecursiveCharacterTextSplitter
    - Preserve original metadata (source file, page numbers, etc.) in every chunk
    - Default to CHUNK_SIZE and CHUNK_OVERLAP values from src/config.py

Why text splitting is essential:
    1. LLMs and embedding models have context limits.
    2. Smaller chunks make retrieval much more precise (pinpointing exact answers
       rather than retrieving an entire 50-page book).
    3. Overlap ensures sentences/ideas cut at boundaries are not lost.

Imported by:
    - src/rag_pipeline.py (in Module 6 / Module 10)
"""

from typing import List, Optional
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create and configure an instance of RecursiveCharacterTextSplitter.

    Args:
        chunk_size (int, optional): Maximum characters per chunk.
                                     Defaults to CHUNK_SIZE from config.py.
        chunk_overlap (int, optional): Character overlap between adjacent chunks.
                                       Defaults to CHUNK_OVERLAP from config.py.

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter instance.
    """
    size = chunk_size if chunk_size is not None else CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of LangChain Document objects into smaller chunk Document objects.

    Args:
        documents (List[Document]): The list of Document objects from Module 2.
        chunk_size (int, optional): Maximum characters per chunk.
        chunk_overlap (int, optional): Character overlap between adjacent chunks.

    Returns:
        List[Document]: List of smaller chunked Document objects with preserved metadata.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks
