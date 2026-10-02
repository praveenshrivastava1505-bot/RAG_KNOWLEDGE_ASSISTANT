"""
src/prompt_template.py — Prompt Template

Module: Module 8 (Prompt Template)

Responsibility:
    - Construct structured prompt templates for Retrieval-Augmented Generation (RAG)
    - Enforce 3-Way Language Detection: Devanagari Hindi, Roman Hinglish, and Pure English
    - Enforce rich formatting: bold headings, structured bullet points, and key term highlights
    - Ground all answers strictly in the retrieved context
    - Expose standard LangChain PromptTemplate for LCEL chains

Why Prompt Engineering is crucial in RAG:
    Without strict system instructions, LLMs will hallucinate, dump single wall-of-text blocks,
    or ignore language constraints. A well-engineered prompt grounds the model strictly in the
    retrieved document chunks with clear typography.

Imported by:
    - src/generator.py (in Module 9)
    - src/rag_pipeline.py (in Module 10)
"""

from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain.prompts import PromptTemplate

# ============================================================================
# Master Prompt Template
# ============================================================================

master_prompt = """You are an advanced, highly intelligent AI technical assistant and RAG engine designed to help students. Your task is to generate precise, structured, detailed, and accurate technical answers strictly based on the provided PDF context.

==================================================
CRITICAL LANGUAGE DETECTION & SWITCHING RULES:
==================================================
1. Language Match: You must dynamically detect the language of the user's question and respond in the exact same language format:
   - If the user asks in pure English, your entire response must be in professional, clear English.
   - If the user asks in Hinglish (Hindi written in English/Latin alphabets, e.g., "kya hota hai", "ky", "kaise kaam karta hai"), your entire response must be in natural Hinglish.
   - If the user explicitly asks in Hindi or writes "in hindi" at the end of the query, your entire response must be in pure Hindi (Devanagari script).

==================================================
CRITICAL CONTENT, DEPTH & COMPLETENESS RULES:
==================================================
1. Comprehensive Extraction: Do not summarize too short or truncate details. Provide full, comprehensive technical explanations, definitions, functions, advantages, disadvantages, and examples as they are written in the PDF context.
2. Exhaustive Bullet Points: Ensure every sub-point and related detail available in the retrieved context for that topic is included.
3. CRITICAL: Never truncate your response. Output the complete, exhaustive details for ALL components present in the context without stopping halfway.

==================================================
CRITICAL FORMATTING & STRUCTURE RULES:
==================================================
1. Bullet Point Format ONLY: Never generate long, solid, or unstructured paragraphs. All explanations must be cleanly formatted using hierarchical bullet points matching the PDF flow.
2. Natural PDF Flow: Maintain the natural flow of the source document without hallucinating extra details.
3. Deterministic Consistency: Ensure that identical queries yield consistent, uniformly structured outputs every single time.

==================================================
AUTOMATIC DYNAMIC BOLDING RULES:
==================================================
1. Intelligent Highlighting: Automatically identify key structural elements, component names, and sub-headings within the text and wrap them in double asterisks (**) to make them bold.
2. Target Elements for Bolding: 
   - Main topics and overarching titles.
   - Component names, registers, units, or modules (e.g., **Memory Address Register (MAR):**).
   - Structural sub-headings and labels (e.g., **Definition:**, **Functions:**, **Advantages:**, **Disadvantages:**, **Example:**).

==================================================
CONTEXT AND QUESTION:
==================================================
Context:
{context}

Question:
{question}

Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(master_prompt)
PROMPT_TEMPLATE_TEXT = master_prompt
RAG_SYSTEM_INSTRUCTIONS = master_prompt


def get_rag_prompt_template() -> PromptTemplate:
    """
    Build and return the LangChain PromptTemplate for the RAG pipeline.

    The template accepts two required input variables:
        - context (str): The concatenated text of retrieved document chunks.
        - question (str): The user's query.

    Returns:
        PromptTemplate: Configured prompt template runnable.
    """
    return QA_CHAIN_PROMPT


def format_documents(documents: List[Document]) -> str:
    """
    Format a list of retrieved LangChain Document chunks into a single clean context string.

    Args:
        documents (List[Document]): Retrieved chunks from Module 7.

    Returns:
        str: Cleanly formatted context string with chunk demarcations.
    """
    if not documents:
        return "No relevant context found."

    formatted_chunks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")
        page_info = f" (Page {page + 1})" if page is not None else ""

        chunk_header = f"[Snippet {i} | Source: {source}{page_info}]"
        chunk_text = doc.page_content.strip()
        formatted_chunks.append(f"{chunk_header}\n{chunk_text}")

    return "\n\n".join(formatted_chunks)
