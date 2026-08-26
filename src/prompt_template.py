"""
src/prompt_template.py — Prompt Template

Module: Module 8 (Prompt Template)

Responsibility:
    - Construct structured prompt templates for Retrieval-Augmented Generation (RAG)
    - Enforce hallucination control: instruct the LLM to answer strictly from retrieved context
    - Instruct the model to say "I don't know based on the provided context" if the answer is missing
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
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# SYSTEM PROMPT INSTRUCTIONS FOR GROUNDED RAG
# ============================================================================

RAG_SYSTEM_INSTRUCTIONS = """You are a helpful, precise, and factual Knowledge Assistant.
Your task is to answer the user's question based ONLY on the provided context snippets below.

Strict Guidelines:
1. Rely solely on the provided Context. Do NOT use outside knowledge or make unsupported assumptions.
2. If the answer cannot be found in or directly deduced from the Context, respond truthfully with:
   "I don't know based on the provided context."
3. Do not attempt to fabricate, guess, or hallucinate an answer.
4. Keep your answer factual, clear, and well-structured.

Context:
{context}"""


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
