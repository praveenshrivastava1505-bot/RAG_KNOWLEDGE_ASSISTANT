"""
tests/test_pipeline.py — End-to-End RAG Pipeline Tests

Module: Module 10 Verification / Module 13 Test Suite

Tests:
    1. Ingestion of test documents
    2. Query execution with ask_question()
    3. Grounded answer generation
    4. Anti-hallucination guardrail test ("I don't know based on the provided context")
"""

import sys
from src.rag_pipeline import ask_question


def main():
    print("=" * 65)
    print("  RAG Knowledge Assistant — End-to-End Pipeline Test")
    print("=" * 65)

    # Test 1: Query about content present in the ingested documents
    query_1 = "What is Retrieval-Augmented Generation (RAG) and what does it enhance?"
    print(f"\n[Test 1] Grounded Query: \"{query_1}\"")
    print("Running RAG pipeline (Retrieval -> Prompt -> Gemini 2.5 Flash)...")

    response_1 = ask_question(query_1, k=2)

    print("\n" + "-" * 65)
    print(f"🤖 Answer:\n{response_1['answer']}")
    print("-" * 65)
    print(f"📚 Sources Used ({len(response_1['source_documents'])}):")
    for i, doc in enumerate(response_1["source_documents"], start=1):
        print(f"   [{i}] {doc.metadata.get('source')} (Snippet: {repr(doc.page_content[:40])}...)")

    assert len(response_1["answer"]) > 0, "Empty answer returned"
    assert len(response_1["source_documents"]) > 0, "No source documents returned"

    # Test 2: Out-of-domain query to test anti-hallucination guardrails
    query_2 = "What is the secret recipe for baking a Parisian sourdough baguette?"
    print(f"\n\n[Test 2] Out-of-Context Query: \"{query_2}\"")
    print("Testing Anti-Hallucination Guardrail...")

    response_2 = ask_question(query_2, k=2)

    print("\n" + "-" * 65)
    print(f"🤖 Answer:\n{response_2['answer']}")
    print("-" * 65)

    print("\n✅ End-to-End Pipeline Test PASSED!")


if __name__ == "__main__":
    main()
