# RAG-Based Knowledge Assistant

A Retrieval-Augmented Generation (RAG) based knowledge assistant that allows users to upload documents, ask questions, and receive answers grounded in the uploaded content.

## Tech Stack

- **Python 3.11+**
- **LangChain** — Document loading, text splitting, chaining
- **Google Gemini** — LLM (gemini-2.5-flash) + Embeddings (gemini-embedding-001)
- **ChromaDB** — Local vector database
- **Streamlit** — Chat UI

## Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your Gemini API key
6. Get your API key from [Google AI Studio](https://aistudio.google.com/)

## Usage

### Ingest documents
```bash
python ingest.py data/sample.pdf
```

### Run the app
```bash
streamlit run app.py
```

## Project Structure

```
rag-knowledge-assistant/
├── data/                    # Sample documents
├── chroma_db/               # Vector database (auto-created)
├── src/                     # Source modules
│   ├── config.py            # Configuration
│   ├── document_loader.py   # Document loading
│   ├── text_splitter.py     # Text chunking
│   ├── embeddings.py        # Gemini embeddings
│   ├── vector_store.py      # ChromaDB operations
│   ├── retriever.py         # Similarity search
│   ├── prompt_template.py   # Prompt construction
│   ├── generator.py         # Gemini LLM
│   └── rag_pipeline.py      # Orchestrator
├── tests/                   # Tests
├── app.py                   # Streamlit UI
├── ingest.py                # CLI ingestion
├── requirements.txt         # Dependencies
└── .env.example             # Environment template
```

## Team

Built as a college minor project (B.Tech AIML, 2nd Year).

## License

MIT
