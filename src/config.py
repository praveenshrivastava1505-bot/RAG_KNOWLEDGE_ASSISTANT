"""
src/config.py — Configuration & Environment Variables

Module: Module 1 (Project Setup & Configuration)

This file is the SINGLE SOURCE OF TRUTH for all project settings.
Every other module imports its configuration from here.

What it does:
    1. Loads your secret API key from the .env file (so it never appears in code)
    2. Defines which models to use (local HuggingFace for embeddings, Hugging Face Inference API for LLM)
    3. Defines how to split documents (chunk size, overlap)
    4. Defines where ChromaDB stores its data

Imported by:
    - src/embeddings.py   (uses EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    - src/vector_store.py (uses CHROMA_DB_DIR, CHROMA_COLLECTION_NAME)
    - src/generator.py    (uses LLM_MODEL)
    - src/rag_pipeline.py (uses various settings)
"""

import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================
# load_dotenv() reads the .env file in the project root and
# makes its variables available via os.getenv().
# This keeps your API key OUT of your code and OUT of GitHub.
# ============================================================

load_dotenv()

# ============================================================
# 2. API KEY
# ============================================================
# Hugging Face Hub token for accessing the Hugging Face Inference API.
# Checks environment variables and Streamlit secrets.
#
# IMPORTANT: Never hardcode your actual key here!
# ============================================================

def _resolve_hf_token() -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        try:
            import streamlit as st
            if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
                token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        except Exception:
            pass
    return token or ""

HUGGINGFACEHUB_API_TOKEN = _resolve_hf_token()

# ============================================================
# 3. MODEL SETTINGS
# ============================================================
# EMBEDDING_MODEL:
#   Local HuggingFace embedding model ("all-MiniLM-L6-v2").
#   Runs locally on your device without API rate limits, daily quotas, or network latency.
#
# EMBEDDING_DIMENSIONS:
#   all-MiniLM-L6-v2 produces dense 384-dimensional semantic vectors.
#
# LLM_MODEL:
#   Fast, open-source model optimized for text-generation and RAG.
#   "HuggingFaceH4/zephyr-7b-beta" works natively with task="text-generation".
#
# LLM_TEMPERATURE:
#   Controls how "creative" the LLM responses are.
#   0.0 = very deterministic (same input → same output)
#   1.0 = very creative (more randomness)
#   0.3 = good balance for RAG (factual but not robotic)
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# ============================================================
# 4. TEXT SPLITTING SETTINGS
# ============================================================
# When we split a document into chunks, we need to decide:
#
# CHUNK_SIZE:
#   Maximum number of characters in each chunk.
#   1000 characters ≈ 150-200 words ≈ a decent paragraph.
#   Too small → loses context. Too large → wastes tokens.
#
# CHUNK_OVERLAP:
#   Number of characters that overlap between adjacent chunks.
#   This prevents information from being "cut in half" at
#   chunk boundaries. 200 chars of overlap is a safe default.
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================================
# 5. CHROMADB SETTINGS
# ============================================================
# CHROMA_DB_DIR:
#   The folder where ChromaDB stores its data on disk.
#   We use a path relative to the project root.
#
# CHROMA_COLLECTION_NAME:
#   The name of the "collection" (like a table) inside ChromaDB
#   where we store our document embeddings.
# ============================================================

CHROMA_DB_DIR = str(Path(__file__).parent.parent / "chroma_db")
CHROMA_COLLECTION_NAME = "knowledge_base"
