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
EMBEDDING_BATCH_SIZE = 32
MAX_EMBEDDING_RETRIES = 3

# Vector search configuration
VECTOR_TOP_K = 5
VECTOR_SIMILARITY_THRESHOLD = 0.7

# Knowledge graph configuration
KG_EXTRACTION_MODE = "heuristic"  # "heuristic" or "llm"
KG_KEYWORDS = {
    "problems": ["problem:", "issue:", "challenge:", "difficulty:", "concern:"],
    "solutions": ["solution:", "resolution:", "fix:", "approach:", "method:"],
    "causes": ["cause:", "reason:", "due to:", "because:", "root cause:"],
    "results": ["result:", "outcome:", "consequence:", "effect:", "impact:"]
}

# Text processing configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100

# Conversation configuration
MAX_CONVERSATION_HISTORY = 10
MAX_CONTEXT_LENGTH = 4000

# LLM configuration
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000
LLM_TIMEOUT = 30

# UI configuration
GRADIO_PORT = 7860
GRADIO_SHARE = False

# Logging configuration
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%^(asctime^)s - %^(name^)s - %^(levelname^)s - %^(message^)s"

# Configure logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STORAGE_ROOT / "chatbot.log")
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
ECHO is off.
    if LLM_PROVIDER not in ["gemini", "openai"]:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be 'gemini' or 'openai'")
ECHO is off.
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY required when LLM_PROVIDER=gemini")
ECHO is off.
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")

# Validate on import
validate_config()
