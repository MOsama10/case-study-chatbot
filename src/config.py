"""
Configuration module for Legal Document AI Assistant.
Optimized for arbitration cases and legal document analysis with complete responses.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"
DATA_ROOT = PROJECT_ROOT / "data"

# Storage directories
INDEXES_DIR = STORAGE_ROOT / "indexes"
KG_DIR = STORAGE_ROOT / "kg"
META_DIR = STORAGE_ROOT / "meta"

# Create directories if they don't exist
for directory in [STORAGE_ROOT, INDEXES_DIR, KG_DIR, META_DIR, DATA_ROOT]:
    directory.mkdir(exist_ok=True, parents=True)

# -----------------------------
# API Configuration
# -----------------------------
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# Model names
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------
# LLM Configuration - Enhanced for Complete Responses
# -----------------------------
LLM_TEMPERATURE = 0.2  # Lower for precise legal responses
LLM_MAX_TOKENS = 4000  # Increased for complete legal analysis
LLM_TIMEOUT = 60  # Increased timeout for longer responses
LLM_RETRY_ATTEMPTS = 2

# Legal-specific LLM settings
LEGAL_LLM_TEMPERATURE = 0.2
LEGAL_LLM_MAX_TOKENS = 4000
LEGAL_SYSTEM_ROLE = "legal_document_analyst"

# -----------------------------
# Embedding Configuration
# -----------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_CACHE_SIZE = 1000
MAX_EMBEDDING_RETRIES = 2

# -----------------------------
# Vector Search Configuration
# -----------------------------
VECTOR_TOP_K = 8  # Number of similar documents to retrieve
VECTOR_SIMILARITY_THRESHOLD = 0.65  # Minimum similarity score
FAISS_SEARCH_PARAMS = {
    'nprobe': 8,
    'efSearch': 16
}

# -----------------------------
# Knowledge Graph Configuration
# -----------------------------
KG_EXTRACTION_MODE = "heuristic"
KG_KEYWORDS = {
    "legal_concepts": ["arbitration:", "clause:", "agreement:", "procedure:", "decision:"],
    "problems": ["problem:", "issue:", "challenge:", "dispute:", "conflict:"],
    "solutions": ["solution:", "resolution:", "remedy:", "approach:", "method:"],
    "results": ["result:", "outcome:", "award:", "ruling:", "judgment:"],
    "procedures": ["step:", "process:", "timeline:", "deadline:", "requirement:"]
}
KG_MAX_NODES_PER_QUERY = 50
KG_SIMILARITY_THRESHOLD = 0.3
KG_MAX_DEPTH = 2

# -----------------------------
# Text Processing Configuration
# -----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 100
MAX_CHUNK_LENGTH = 1200

# Document processing
DOC_PROCESSING_BATCH_SIZE = 5
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKER_THREADS = 4

# -----------------------------
# Context and Conversation Configuration
# -----------------------------
MAX_CONVERSATION_HISTORY = 10
MAX_CONTEXT_SOURCES = 8
MAX_CONTEXT_LENGTH = 8000  # Increased for comprehensive legal context

# -----------------------------
# Response Configuration - Enhanced for Legal Documents
# -----------------------------
# Professional mode settings
PROFESSIONAL_MODE = True
BUSINESS_LANGUAGE_LEVEL = "legal"  # Focused on legal terminology

# Response length limits
MIN_RESPONSE_LENGTH = 300  # Minimum for comprehensive legal analysis
MAX_RESPONSE_LENGTH = 4000  # Maximum for detailed responses
LEGAL_RESPONSE_MIN_LENGTH = 300
LEGAL_RESPONSE_MAX_LENGTH = 4000

# Legal formatting options
USE_LEGAL_CITATIONS = True
INCLUDE_LEGAL_DISCLAIMERS = True
VALIDATE_LEGAL_RESPONSE_QUALITY = True
AUTO_FORMAT_LEGAL_STRUCTURE = True

# -----------------------------
# Direct Answer Configuration
# -----------------------------
ENABLE_DIRECT_ANSWERS = True
DIRECT_ANSWER_CONFIDENCE_THRESHOLD = 0.6
DIRECT_ANSWER_MAX_LENGTH = 1000  # Increased for legal direct answers
DIRECT_ANSWER_MIN_EVIDENCE = 2

# Legal question patterns
FACTUAL_QUESTION_PATTERNS = [
    r'^(what|when|where|who|which|how\s+much|how\s+many|how\s+long)',
    r'^(define|explain|describe|clarify)\s+',
    r'(timeline|deadline|cost|fee|duration|procedure)',
    r'^(is\s+there|are\s+there|does\s+the|can\s+the)'
]

# Question classification
DIRECT_ANSWER_TRIGGERS = {
    'question_starters': ['is', 'are', 'was', 'were', 'will', 'would', 'could', 'should', 'can', 'do', 'does', 'did'],
    'legal_indicators': ['clause', 'agreement', 'contract', 'arbitration', 'procedure', 'timeline', 'cost'],
    'factual_indicators': ['what is', 'who is', 'when did', 'where is', 'how much', 'how long'],
    'binary_indicators': ['yes or no', 'true or false', 'valid or invalid'],
    'max_word_count': 15
}

# -----------------------------
# Retrieval Configuration
# -----------------------------
HYBRID_SEARCH_WEIGHTS = {
    'vector': 0.6,
    'kg': 0.3,
    'keyword': 0.1
}

# Legal-specific retrieval
LEGAL_CONFIDENCE_THRESHOLD = 0.4
LEGAL_SOURCE_COUNT = 6

# -----------------------------
# Caching Configuration
# -----------------------------
ENABLE_QUERY_CACHE = True
QUERY_CACHE_SIZE = 100
EMBEDDING_CACHE_TTL = 3600  # 1 hour

# -----------------------------
# Quality Thresholds
# -----------------------------
MIN_RESPONSE_CONFIDENCE = 0.3
MIN_SOURCE_RELEVANCE = 0.4
MAX_RESPONSE_TIME = 45  # seconds
QUALITY_THRESHOLD = 0.6

# -----------------------------
# UI Configuration
# -----------------------------
GRADIO_PORT = 7860
GRADIO_SHARE = False
GRADIO_QUEUE_SIZE = 10
GRADIO_MAX_THREADS = 4
GRADIO_ENABLE_STREAMING = True

# -----------------------------
# Logging Configuration
# -----------------------------
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STORAGE_ROOT / "legal_chatbot.log", encoding="utf-8")
    ]
)

# -----------------------------
# Legal Document Specific Settings
# -----------------------------
# Arbitration-specific keywords for enhanced analysis
ARBITRATION_KEYWORDS = [
    'arbitration', 'arbitral', 'arbitrator', 'tribunal', 'award',
    'clause', 'agreement', 'procedure', 'hearing', 'proceeding',
    'jurisdiction', 'seat', 'venue', 'rules', 'institution',
    'emergency', 'interim', 'provisional', 'measures', 'relief',
    'costs', 'fees', 'expenses', 'timeline', 'deadline',
    'enforcement', 'recognition', 'challenge', 'annulment',
    'validity', 'scope', 'applicability', 'jurisdiction'
]

# Legal document types for classification
LEGAL_DOCUMENT_TYPES = [
    'contract', 'agreement', 'clause', 'treaty', 'convention',
    'rules', 'procedure', 'guideline', 'statute', 'regulation',
    'case', 'precedent', 'decision', 'award', 'ruling'
]

# -----------------------------
# Response Templates for Legal Content
# -----------------------------
LEGAL_RESPONSE_STYLES = {
    'analysis': {
        'introduction': 'Based on the legal documents and arbitration precedents:',
        'conclusion': 'This analysis is based on available legal documentation.',
        'structure': ['Legal Framework', 'Analysis', 'Conclusion', 'Recommendations']
    },
    'direct_answer': {
        'affirmative': 'Yes, based on the legal documents:',
        'negative': 'No, according to the legal documentation:',
        'uncertain': 'The legal documents indicate uncertainty regarding:'
    },
    'procedural': {
        'timeline': 'According to the arbitration procedures:',
        'requirements': 'The legal requirements include:',
        'steps': 'The procedural steps are:'
    }
}

# -----------------------------
# Helper Functions
# -----------------------------
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


def validate_config() -> None:
    """Validate configuration and raise errors for missing required settings."""
    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        raise ValueError(
            "No API key found. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file"
        )

    if LLM_PROVIDER not in ["gemini", "openai"]:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be 'gemini' or 'openai'")

    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY required when LLM_PROVIDER=gemini")

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")

    # Validate token limits
    if LLM_MAX_TOKENS < 1000:
        raise ValueError("LLM_MAX_TOKENS should be at least 1000 for complete responses")

    # Validate paths
    if not PROJECT_ROOT.exists():
        raise ValueError(f"Project root directory not found: {PROJECT_ROOT}")


# Validate configuration on import
validate_config()

# Log configuration status
logger = get_logger(__name__)
logger.info(f"Legal Document AI Assistant configuration loaded")
logger.info(f"LLM Provider: {LLM_PROVIDER}")
logger.info(f"Max Tokens: {LLM_MAX_TOKENS}")
logger.info(f"Professional Mode: {PROFESSIONAL_MODE}")
logger.info(f"Legal Citations: {USE_LEGAL_CITATIONS}")