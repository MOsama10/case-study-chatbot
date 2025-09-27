# Case Study Conversational Chatbot PoC

A professional Proof-of-Concept implementation of an AI-powered chatbot that analyzes case studies using semantic search, knowledge graphs, and agentic RAG (Retrieval-Augmented Generation).

## Features

- **Document Processing**: Parses Word documents with case studies and Q&A pairs
- **Semantic Search**: Uses sentence transformers and FAISS for efficient similarity search
- **Knowledge Graph**: Builds relationships between problems, solutions, and case studies
- **Agentic RAG**: Intelligent query classification, retrieval, and LLM-powered responses
- **Conversational UI**: Gradio-based chat interface with source citations

## System Requirements

- Windows 10/11 (tested on Windows 10 Pro 22H2)
- Python 3.10-3.11
- 16GB RAM recommended
- CPU-only operation (no GPU required)

## Quick Start

### 1. Setup Environment
```cmd
# Navigate to project directory
cd case-study-chatbot

# Activate virtual environment
venv\Scripts\activate
```

### 2. Configure API Keys
Edit the `.env` file:
```env
# Choose one:
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Your Data
Place your case study document at `data/cases.docx`

### 4. Run the Application
```cmd
python src/ui.py
```

The Gradio interface will open at `http://localhost:7860`

## Development

### Running Tests
```cmd
pytest tests/
```

## Troubleshooting

### FAISS Installation Issues
If you encounter FAISS installation problems:
```cmd
pip install --no-binary faiss-cpu faiss-cpu
```

### Memory Issues
Reduce batch size in config:
```python
EMBEDDING_BATCH_SIZE = 16  # Default: 32
```

## License

MIT License
