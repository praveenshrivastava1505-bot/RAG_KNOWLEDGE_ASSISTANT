"""
src/prompt_template.py — Prompt Template

Module: Module 8 (Prompt Template)

Responsibility:
    - Construct structured prompt templates for Retrieval-Augmented Generation (RAG)
    - Enforce 3-Way Language Detection: Devanagari Hindi, Roman Hinglish, and Pure English
    - Ground all answers strictly in the retrieved context
    - Expose standard LangChain ChatPromptTemplate for LCEL chains

Why Prompt Engineering is crucial in RAG:
    Without strict system instructions, LLMs will hallucinate or fall back to their
    pre-training data. A well-engineered prompt grounds the model strictly in the
    retrieved document chunks.

Imported by:
    - src/generator.py (in Module 9)
    - src/rag_pipeline.py (in Module 10)
"""

from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# SYSTEM PROMPT INSTRUCTIONS FOR 3-WAY LANGUAGE DETECTION
# ============================================================================

RAG_SYSTEM_INSTRUCTIONS = """You are an expert AI assistant. You must analyze the exact phrasing of the user's question and choose the output language based strictly on these 3 rules:

RULE 1: Explicit Hindi Request -> Devanagari Script
IF the user's question explicitly contains the phrase "in hindi" (e.g., "what is general register organisation in hindi"):
THEN you MUST write the entire answer in proper Devanagari Hindi script (हिंदी).
Example format: "जनरल रजिस्टर ऑर्गनाइजेशन CPU के अंदर रजिस्टर्स की एक व्यवस्था है..."

RULE 2: Hinglish Keywords -> Hinglish (Roman Script)
IF the user's question contains ANY Hinglish conversational words (e.g., "kya hota hai", "samjha do", "kaise", "batao"):
THEN you MUST translate the context and write the entire answer in conversational Hinglish (Hindi language using the English alphabet). Do NOT use Devanagari.
Example format: "General Register Organization CPU ke andar registers ka ek system hota hai jo data store karta hai..."

RULE 3: Pure English -> Pure English
IF the question is completely in pure English with NO Hinglish words and NO "in hindi" request:
THEN you MUST write the entire answer in pure English.
Example format: "General Register Organization refers to the arrangement of registers inside the CPU..."

Always provide well-structured, detailed answers with bullet points based ONLY on the provided context.

Context: {context}"""


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Build and return the LangChain ChatPromptTemplate for the RAG pipeline.

    The template accepts two required input variables:
        - context (str): The concatenated text of retrieved document chunks.
        - question (str): The user's query.

    Returns:
        ChatPromptTemplate: Configured prompt template runnable.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_INSTRUCTIONS),
            ("human", "{question}"),
        ]
    )
    return prompt


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
