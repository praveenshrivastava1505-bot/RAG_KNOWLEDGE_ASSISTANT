"""
src/document_loader.py — Document Loading

Module: Module 2 (Document Loading)

Responsibility:
    - Load PDF, DOCX, and TXT files from disk
    - Extract text content from each file format using LangChain loaders
    - Return a list of LangChain Document objects
    - Each Document contains:
        - page_content: string containing extracted text
        - metadata: dictionary containing file details (source, page number, etc.)

Supported Formats:
    - .pdf  -> PyPDFLoader
    - .docx -> Docx2txtLoader
    - .txt  -> TextLoader (UTF-8 encoding)

Imported by:
    - src/rag_pipeline.py (in Module 6 / Module 10)
"""

import os
from pathlib import Path
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from the specified file path and return its content
    as a list of LangChain Document objects.

    Args:
        file_path (str): Path to the target document (.pdf, .docx, or .txt).

    Returns:
        List[Document]: A list of Document objects with .page_content and .metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported or file path is invalid.
    """
    # 1. Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: '{file_path}'")

    if not os.path.isfile(file_path):
        raise ValueError(f"The path provided is not a regular file: '{file_path}'")

    # 2. Extract file extension (case-insensitive)
    file_extension = Path(file_path).suffix.lower()

    # 3. Select appropriate loader based on file type
    if file_extension == ".pdf":
        # PyPDFLoader reads the PDF and creates one Document per page
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        # Docx2txtLoader extracts text from Word documents
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".txt":
        # TextLoader reads plain text files with explicit UTF-8 encoding
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        supported_extensions = [".pdf", ".docx", ".txt"]
        raise ValueError(
            f"Unsupported file format '{file_extension}'. "
            f"Supported formats are: {', '.join(supported_extensions)}"
        )

    # 4. Load and return documents
    documents = loader.load()
    return documents
