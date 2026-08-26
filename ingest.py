"""
ingest.py — CLI Document Ingestion Script

Module: Module 6 (Document Ingestion Pipeline)

Responsibility:
    - Command-line interface to ingest documents into the ChromaDB vector database
    - Invokes src/rag_pipeline.py ingest_document()
    - Provides real-time feedback in the terminal

Usage:
    python ingest.py data/sample.pdf
    python ingest.py data/sample.txt
"""

import sys
import argparse
from pathlib import Path

from src.rag_pipeline import ingest_document


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG Knowledge Assistant vector store."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default="data/sample.txt",
        help="Path to the document to ingest (.pdf, .docx, .txt). Defaults to data/sample.txt",
    )
    args = parser.parse_args()

    file_path = args.file_path

    print("=" * 60)
    print("  RAG Knowledge Assistant — Document Ingestion CLI")
    print("=" * 60)
    print(f"[*] Target file: {file_path}")

    if not Path(file_path).exists():
        print(f"[!] Error: File not found at '{file_path}'")
        sys.exit(1)

    try:
        print("[1/3] Loading document...")
        print("[2/3] Splitting into text chunks...")
        print("[3/3] Generating embeddings & saving to ChromaDB...")

        result = ingest_document(file_path)

        print("\n" + "-" * 60)
        print("✅ Ingestion Complete!")
        print(f"   - File Ingested     : {result['file_path']}")
        print(f"   - Pages/Docs Read   : {result['raw_pages_loaded']}")
        print(f"   - Chunks Created    : {result['chunks_created']}")
        print(f"   - Total DB Chunks   : {result['total_chunks_in_db']}")
        print("-" * 60 + "\n")

    except Exception as e:
        print(f"\n[!] Ingestion Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
