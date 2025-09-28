# Legal Document AI Assistant

A professional AI-powered assistant specialized in arbitration law and legal document analysis. Built to analyze arbitration cases, legal procedures, decision trees, and provide expert guidance on commercial dispute resolution.

## Features

- **Arbitration Analysis** - Timeline analysis, cost assessments, and procedural guidance
- **Legal Document Processing** - Intelligent parsing of Word documents (.docx) and text files
- **Knowledge Graph Integration** - Structured relationship mapping between legal concepts
- **Semantic Search** - Advanced document retrieval using embeddings and vector similarity
- **Professional UI** - Clean, modern web interface optimized for legal professionals
- **Multi-LLM Support** - Compatible with Google Gemini and OpenAI APIs

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd legal-document-ai-assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your API credentials:

```env
# Choose your AI provider:

# Option 1: Google Gemini (Free tier available)
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini

# Option 2: OpenAI (Paid service)
# OPENAI_API_KEY=your_openai_api_key_here
# LLM_PROVIDER=openai
```

**Get API Keys:**
- [Google Gemini API Key](https://makersuite.google.com/app/apikey) (Free tier)
- [OpenAI API Key](https://platform.openai.com/api-keys) (Paid)

### 3. Launch

```bash
# Simple launch
python run_chatbot.py

# Or use the main launcher
python main.py

# Or launch UI directly
python src/ui.py
```

Open your browser to `http://localhost:7860`

## Document Management

### Adding Legal Documents

1. Place your `.docx` files in the `data/` folder
2. For batch processing, use the `data/batch_1/` subfolder
3. The system automatically processes documents on startup

### Supported File Types

- **Primary**: Microsoft Word (.docx) documents
- **Fallback**: Plain text (.txt) files
- **Focus**: Arbitration agreements, legal procedures, case studies, decision trees

## Core Capabilities

### Legal Analysis
- Arbitration clause validity assessment
- Timeline and procedural guidance
- Cost analysis and fee structures
- Emergency arbitration conditions
- Award enforcement procedures

### Document Intelligence
- Semantic search across legal documents
- Relationship mapping between legal concepts
- Question-answering from document content
- Precedent identification and analysis

### Professional Features
- Confidence scoring for legal advice
- Source attribution and citations
- Session history and context awareness
- Professional legal language and formatting

## Usage Examples

### Arbitration Queries
```
"What is the timeline for arbitration proceedings?"
"How much do arbitration costs typically range?"
"What makes an arbitration clause valid?"
"When is emergency arbitration available?"
```

### Legal Analysis
```
"Compare arbitration costs to claimed amounts"
"Analyze the validity of this arbitration clause"
"Find precedents for emergency arbitration"
"Explain the decision tree for enforcement"
```

## Architecture

### Core Components

- **`src/agent.py`** - Legal counsel AI agent with professional persona
- **`src/embeddings.py`** - Document vectorization and semantic search
- **`src/knowledge_graph.py`** - Legal concept relationship mapping
- **`src/retriever.py`** - Answer-focused document retrieval
- **`src/ui.py`** - Professional web interface
- **`src/chatbot.py`** - Conversation management and context

### Data Processing Pipeline

1. **Document Ingestion** - Load and parse legal documents
2. **Text Processing** - Chunk, clean, and preprocess content
3. **Embedding Generation** - Create semantic vectors using Sentence Transformers
4. **Knowledge Graph** - Build relationship networks between legal concepts
5. **Index Storage** - Save processed data for fast retrieval

## Advanced Setup

### Custom Configuration

Edit `src/config.py` for advanced settings:

```python
# Response configuration
LLM_MAX_TOKENS = 4000           # Detailed legal responses
LEGAL_RESPONSE_MIN_LENGTH = 300 # Comprehensive analysis
USE_LEGAL_CITATIONS = True      # Professional formatting

# Retrieval settings
VECTOR_TOP_K = 8               # Document search results
LEGAL_CONFIDENCE_THRESHOLD = 0.4 # Answer quality threshold
```

### Development Mode

```bash
# Run comprehensive setup with validation
python comprehensive_setup.py

# Run tests
pytest tests/

# Launch with debug logging
python src/ui.py --port 7860 --share
```

## Legal Disclaimer

This tool provides AI-assisted analysis of legal documents for informational purposes. It should not replace professional legal counsel or formal legal advice. Always consult qualified legal professionals for official legal matters.

## Technical Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and indexes
- **Network**: Internet connection for LLM API calls

## Support

- **Documentation**: Review source code comments for detailed implementation notes
- **Configuration**: Check `src/config.py` for all available settings
- **Troubleshooting**: See log files in `storage/` directory for debugging

---

**Legal Document AI Assistant** - Professional arbitration and legal document analysis powered by advanced AI.
