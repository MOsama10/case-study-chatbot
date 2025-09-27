"""
Configuration module for Case Study Chatbot.
Loads environment variables and defines storage paths and constants.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
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

# API Configuration
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Performance optimizations
EMBEDDING_BATCH_SIZE = 16  # Reduced for stability
EMBEDDING_CACHE_SIZE = 1000  # Cache recent embeddings
MAX_EMBEDDING_RETRIES = 2

# Model names (use full model path for Gemini)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Vector search configuration
VECTOR_TOP_K = 8  # Increased for better recall
VECTOR_SIMILARITY_THRESHOLD = 0.65  # Slightly lower for more results
FAISS_SEARCH_PARAMS = {
    'nprobe': 8,      # FAISS search parameter
    'efSearch': 16    # For HNSW indexes
}

# Knowledge graph configuration
KG_EXTRACTION_MODE = "heuristic"  # "heuristic" or "llm"
KG_KEYWORDS = {
    "problems": ["problem:", "issue:", "challenge:", "difficulty:", "concern:"],
    "solutions": ["solution:", "resolution:", "fix:", "approach:", "method:"],
    "causes": ["cause:", "reason:", "due to:", "because:", "root cause:"],
    "results": ["result:", "outcome:", "consequence:", "effect:", "impact:"]
}
KG_MAX_NODES_PER_QUERY = 50
KG_SIMILARITY_THRESHOLD = 0.3
KG_MAX_DEPTH = 2

# Text processing configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 100
MAX_CHUNK_LENGTH = 1200

# Conversation configuration
MAX_CONVERSATION_HISTORY = 10
MAX_CONTEXT_SOURCES = 8
MAX_CONTEXT_LENGTH = 6000

# LLM configuration
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT = 45
LLM_RETRY_ATTEMPTS = 2

# Retrieval optimizations
HYBRID_SEARCH_WEIGHTS = {
    'vector': 0.6,
    'kg': 0.3,
    'keyword': 0.1
}

# Caching settings
ENABLE_QUERY_CACHE = True
QUERY_CACHE_SIZE = 100
EMBEDDING_CACHE_TTL = 3600  # 1 hour

# UI configuration
GRADIO_PORT = 7860
GRADIO_SHARE = False
GRADIO_QUEUE_SIZE = 10
GRADIO_MAX_THREADS = 4
GRADIO_ENABLE_STREAMING = True

# Document processing optimizations
DOC_PROCESSING_BATCH_SIZE = 5
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKER_THREADS = 4

# Quality thresholds
MIN_RESPONSE_CONFIDENCE = 0.3
MIN_SOURCE_RELEVANCE = 0.4
MAX_RESPONSE_TIME = 30  # seconds

# Logging configuration
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Professional Response Configuration
PROFESSIONAL_MODE = True
MIN_RESPONSE_LENGTH = 200
MAX_RESPONSE_LENGTH = 3000
QUALITY_THRESHOLD = 0.7
BUSINESS_LANGUAGE_LEVEL = "executive"

# Enhanced LLM Parameters for Professional Responses
LLM_TEMPERATURE = 0.3  # Lower for more focused responses
LLM_MAX_TOKENS = 2500  # Increased for comprehensive responses
LLM_SYSTEM_ROLE = "senior_business_analyst"

# Professional Formatting Options
USE_EXECUTIVE_SUMMARIES = True
INCLUDE_METHODOLOGY_FOOTER = True
VALIDATE_RESPONSE_QUALITY = True
AUTO_FORMAT_STRUCTURE = True

# Configure logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STORAGE_ROOT / "chatbot.log", encoding="utf-8")
    ]
)


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
        if "gemini" not in GEMINI_MODEL:
            raise ValueError(
                f"Unsupported GEMINI_MODEL: {GEMINI_MODEL}. Must contain 'gemini' (e.g., 'models/gemini-pro-latest')"
            )

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
        if not OPENAI_MODEL.startswith("gpt-"):
            raise ValueError(
                f"Unsupported OPENAI_MODEL: {OPENAI_MODEL}. Must start with 'gpt-'"
            )


# Validate on import
validate_config()
