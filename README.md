# VE-RAG-System

Agentic RAG (Retrieval-Augmented Generation) System for NVIDIA DGX Spark.

## Features

- **Document Processing**: Docling-powered parsing with OCR, table extraction, and semantic chunking
- **Vector Storage**: ChromaDB for persistent vector storage
- **LLM Integration**: Ollama with configurable models (qwen3:8b default)
- **Web UI**: Gradio interface on port 8501
- **Multi-format Support**: PDF, DOCX, XLSX, PPTX, HTML, TXT, CSV, and more

## Requirements

- Python 3.12+
- Ollama running locally
- NVIDIA GPU (for optimal performance)

## Quick Start

```bash
# First time setup
./setup.sh

# Start the server
./start.sh

# Access at http://localhost:8501
```

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Document Parsing | Docling (local ML) | Layout analysis, table detection, OCR |
| Embeddings | nomic-embed-text | Vector embeddings for semantic search |
| Chat/Reasoning | qwen3:8b | Query routing, RAG responses, evaluation |

## Configuration

Environment variables (set in `start.sh` or export before running):

- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `EMBEDDING_MODEL`: Embedding model name (default: nomic-embed-text)
- `CHAT_MODEL`: Chat model name (default: qwen3:8b)
- `CHROMA_PERSIST_DIR`: ChromaDB storage path (default: ./chroma_db)

## License

Internal use only.
