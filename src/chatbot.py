# """
# Conversation session manager for the case study chatbot.
# Maintains conversation history and context.
# """

# from typing import Dict, List, Any, Optional
# import logging
# import time
# import json
# from datetime import datetime
# from dataclasses import dataclass, asdict

# from src.config import get_logger, MAX_CONVERSATION_HISTORY, MAX_CONTEXT_LENGTH
# from src.agent import get_agent, answer_query

# logger = get_logger(__name__)


# @dataclass
# class ConversationTurn:
#     """Represents a single conversation turn."""
#     timestamp: str
#     user_message: str
#     bot_response: str
#     sources: List[Dict[str, Any]]
#     query_type: str
#     confidence: float
#     processing_time: float


# class ConversationSession:
#     """Manages a single conversation session."""
    
#     def __init__(self, user_id: str):
#         """Initialize conversation session."""
#         self.user_id = user_id
#         self.session_id = f"{user_id}_{int(time.time())}"
#         self.history: List[ConversationTurn] = []
#         self.created_at = datetime.now().isoformat()
#         self.last_activity = self.created_at
        
#         logger.info(f"Created new conversation session: {self.session_id}")
    
#     def add_turn(self, user_message: str, bot_response: Dict[str, Any]) -> None:
#         """Add a conversation turn to history."""
#         turn = ConversationTurn(
#             timestamp=datetime.now().isoformat(),
#             user_message=user_message,
#             bot_response=bot_response['answer'],
#             sources=bot_response['sources'],
#             query_type=bot_response['query_type'],
#             confidence=bot_response['confidence'],
#             processing_time=bot_response['processing_time']
#         )
        
#         self.history.append(turn)
#         self.last_activity = turn.timestamp
        
#         # Maintain history limit
#         if len(self.history) > MAX_CONVERSATION_HISTORY:
#             self.history = self.history[-MAX_CONVERSATION_HISTORY:]
        
#         logger.debug(f"Added turn to session {self.session_id}. History length: {len(self.history)}")
    
#     def get_context(self) -> str:
#         """
#         Get conversation context for the LLM.
        
#         Returns:
#             Formatted conversation history
#         """
#         if not self.history:
#             return ""
        
#         context_parts = []
#         total_length = 0
        
#         # Add recent turns first (reverse chronological)
#         for turn in reversed(self.history[-5:]):  # Last 5 turns
#             turn_text = f"User: {turn.user_message}\nAssistant: {turn.bot_response[:200]}..."
            
#             if total_length + len(turn_text) > MAX_CONTEXT_LENGTH:
#                 break
            
#             context_parts.insert(0, turn_text)
#             total_length += len(turn_text)
        
#         return "\n\n".join(context_parts)
    
#     def get_summary(self) -> Dict[str, Any]:
#         """Get session summary."""
#         if not self.history:
#             return {
#                 'session_id': self.session_id,
#                 'user_id': self.user_id,
#                 'total_turns': 0,
#                 'created_at': self.created_at,
#                 'last_activity': self.last_activity
#             }
        
#         # Calculate statistics
#         query_types = [turn.query_type for turn in self.history]
#         avg_confidence = sum(turn.confidence for turn in self.history) / len(self.history)
#         avg_processing_time = sum(turn.processing_time for turn in self.history) / len(self.history)
        
#         return {
#             'session_id': self.session_id,
#             'user_id': self.user_id,
#             'total_turns': len(self.history),
#             'created_at': self.created_at,
#             'last_activity': self.last_activity,
#             'query_types': list(set(query_types)),
#             'avg_confidence': round(avg_confidence, 2),
#             'avg_processing_time': round(avg_processing_time, 2)
#         }
    
#     def export_history(self) -> List[Dict[str, Any]]:
#         """Export conversation history as JSON-serializable data."""
#         return [asdict(turn) for turn in self.history]


# class ChatbotManager:
#     """Manages multiple conversation sessions."""
    
#     def __init__(self):
#         """Initialize chatbot manager."""
#         self.sessions: Dict[str, ConversationSession] = {}
#         self.agent = get_agent()
        
#         logger.info("Chatbot manager initialized")
    
#     def get_or_create_session(self, user_id: str) -> ConversationSession:
#         """Get existing session or create new one."""
#         if user_id not in self.sessions:
#             self.sessions[user_id] = ConversationSession(user_id)
        
#         return self.sessions[user_id]
    
#     def handle_message(self, user_id: str, message: str) -> Dict[str, Any]:
#         """
#         Handle incoming user message.
        
#         Args:
#             user_id: Unique identifier for the user
#             message: User's message
            
#         Returns:
#             Bot response with metadata
#         """
#         try:
#             logger.info(f"Processing message from user {user_id}: {message[:100]}...")
            
#             # Get or create session
#             session = self.get_or_create_session(user_id)
            
#             # Prepare context from conversation history
#             conversation_context = {
#                 'history': session.get_context(),
#                 'session_id': session.session_id,
#                 'turn_number': len(session.history) + 1
#             }
            
#             # Get response from agent
#             agent_response = answer_query(message, conversation_context)
            
#             # Add to session history
#             session.add_turn(message, agent_response)
            
#             # Prepare response for UI
#             response = {
#                 'answer': agent_response['answer'],
#                 'sources': agent_response['sources'],
#                 'kg_nodes': agent_response['kg_nodes'],
#                 'query_type': agent_response['query_type'],
#                 'confidence': agent_response['confidence'],
#                 'processing_time': agent_response['processing_time'],
#                 'session_info': {
#                     'session_id': session.session_id,
#                     'turn_number': len(session.history),
#                     'total_turns': len(session.history)
#                 }
#             }
            
#             logger.info(f"Message processed successfully for user {user_id}")
#             return response
            
#         except Exception as e:
#             logger.error(f"Error handling message for user {user_id}: {e}")
            
#             return {
#                 'answer': "I apologize, but I encountered an error processing your message. Please try again.",
#                 'sources': [],
#                 'kg_nodes': [],
#                 'query_type': 'error',
#                 'confidence': 0.0,
#                 'processing_time': 0.0,
#                 'session_info': {
#                     'error': str(e)
#                 }
#             }
    
#     def get_session_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
#         """Get summary of user's session."""
#         if user_id in self.sessions:
#             return self.sessions[user_id].get_summary()
#         return None
    
#     def export_session(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
#         """Export user's conversation history."""
#         if user_id in self.sessions:
#             return self.sessions[user_id].export_history()
#         return None
    
#     def clear_session(self, user_id: str) -> bool:
#         """Clear user's conversation session."""
#         if user_id in self.sessions:
#             del self.sessions[user_id]
#             logger.info(f"Cleared session for user {user_id}")
#             return True
#         return False
    
#     def get_active_sessions(self) -> List[str]:
#         """Get list of active session user IDs."""
#         return list(self.sessions.keys())


# # Global chatbot manager instance
# _chatbot_manager = None

# def get_chatbot_manager() -> ChatbotManager:
#     """Get the global chatbot manager instance."""
#     global _chatbot_manager
#     if _chatbot_manager is None:
#         _chatbot_manager = ChatbotManager()
#     return _chatbot_manager


# def handle_message(user_id: str, message: str) -> Dict[str, Any]:
#     """
#     Convenience function to handle messages.
    
#     Args:
#         user_id: User identifier
#         message: User message
        
#     Returns:
#         Bot response
#     """
#     manager = get_chatbot_manager()
#     return manager.handle_message(user_id, message)


# def get_session_info(user_id: str) -> Optional[Dict[str, Any]]:
#     """Get session information for user."""
#     manager = get_chatbot_manager()
#     return manager.get_session_summary(user_id)


# def clear_user_session(user_id: str) -> bool:
#     """Clear user's session."""
#     manager = get_chatbot_manager()
#     return manager.clear_session(user_id)


# if __name__ == "__main__":
#     # For testing
#     test_user = "test_user_1"
    
#     # Test conversation
#     test_messages = [
#         "What are the main problems in customer service?",
#         "How can these problems be solved?",
#         "Can you give me specific examples from the case studies?"
#     ]
    
#     for i, message in enumerate(test_messages, 1):
#         print(f"\n--- Turn {i} ---")
#         print(f"User: {message}")
        
#         response = handle_message(test_user, message)
#         print(f"Bot: {response['answer'][:200]}...")
#         print(f"Sources: {len(response['sources'])}")
#         print(f"Confidence: {response['confidence']:.2f}")
    
#     # Get session summary
#     summary = get_session_info(test_user)
#     print(f"\nSession Summary: {summary}")
################################
# """
# Fixed Configuration module for Legal Document AI Assistant.
# Addresses import and path issues.
# """

# import os
# import logging
# from pathlib import Path
# from typing import Optional
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # -----------------------------
# # Project Paths
# # -----------------------------
# PROJECT_ROOT = Path(__file__).parent.parent
# STORAGE_ROOT = PROJECT_ROOT / "storage"
# DATA_ROOT = PROJECT_ROOT / "data"

# # Storage directories
# INDEXES_DIR = STORAGE_ROOT / "indexes"
# KG_DIR = STORAGE_ROOT / "kg"
# META_DIR = STORAGE_ROOT / "meta"

# # Create directories if they don't exist
# for directory in [STORAGE_ROOT, INDEXES_DIR, KG_DIR, META_DIR, DATA_ROOT]:
#     directory.mkdir(exist_ok=True, parents=True)

# # -----------------------------
# # API Configuration
# # -----------------------------
# GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
# OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
# LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# # Model names
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# # -----------------------------
# # LLM Configuration
# # -----------------------------
# LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
# LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
# LLM_RETRY_ATTEMPTS = 2

# # -----------------------------
# # Embedding Configuration
# # -----------------------------
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
# EMBEDDING_CACHE_SIZE = 1000
# MAX_EMBEDDING_RETRIES = 2

# # -----------------------------
# # Vector Search Configuration
# # -----------------------------
# VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
# VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.65"))
# FAISS_SEARCH_PARAMS = {
#     'nprobe': 8,
#     'efSearch': 16
# }

# # -----------------------------
# # Knowledge Graph Configuration
# # -----------------------------
# KG_EXTRACTION_MODE = "heuristic"
# KG_KEYWORDS = {
#     "legal_concepts": ["arbitration:", "clause:", "agreement:", "procedure:", "decision:"],
#     "problems": ["problem:", "issue:", "challenge:", "dispute:", "conflict:"],
#     "solutions": ["solution:", "resolution:", "remedy:", "approach:", "method:"],
#     "results": ["result:", "outcome:", "award:", "ruling:", "judgment:"],
#     "procedures": ["step:", "process:", "timeline:", "deadline:", "requirement:"]
# }
# KG_MAX_NODES_PER_QUERY = 50
# KG_SIMILARITY_THRESHOLD = 0.3
# KG_MAX_DEPTH = 2

# # -----------------------------
# # Text Processing Configuration
# # -----------------------------
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 150
# MIN_CHUNK_LENGTH = 100
# MAX_CHUNK_LENGTH = 1200

# # Document processing
# DOC_PROCESSING_BATCH_SIZE = 5
# ENABLE_PARALLEL_PROCESSING = True
# MAX_WORKER_THREADS = 4

# # -----------------------------
# # Context and Conversation Configuration
# # -----------------------------
# MAX_CONVERSATION_HISTORY = 10
# MAX_CONTEXT_SOURCES = 8
# MAX_CONTEXT_LENGTH = 6000

# # -----------------------------
# # Response Configuration
# # -----------------------------
# PROFESSIONAL_MODE = True
# BUSINESS_LANGUAGE_LEVEL = "legal"

# MIN_RESPONSE_LENGTH = 200
# MAX_RESPONSE_LENGTH = 3000

# USE_LEGAL_CITATIONS = True
# INCLUDE_LEGAL_DISCLAIMERS = True
# VALIDATE_LEGAL_RESPONSE_QUALITY = True
# AUTO_FORMAT_LEGAL_STRUCTURE = True

# # -----------------------------
# # Direct Answer Configuration
# # -----------------------------
# ENABLE_DIRECT_ANSWERS = True
# DIRECT_ANSWER_CONFIDENCE_THRESHOLD = 0.6
# DIRECT_ANSWER_MAX_LENGTH = 800
# DIRECT_ANSWER_MIN_EVIDENCE = 2

# # -----------------------------
# # Retrieval Configuration
# # -----------------------------
# HYBRID_SEARCH_WEIGHTS = {
#     'vector': 0.6,
#     'kg': 0.3,
#     'keyword': 0.1
# }

# LEGAL_CONFIDENCE_THRESHOLD = 0.4
# LEGAL_SOURCE_COUNT = 6

# # -----------------------------
# # Quality Thresholds
# # -----------------------------
# MIN_RESPONSE_CONFIDENCE = 0.3
# MIN_SOURCE_RELEVANCE = 0.4
# MAX_RESPONSE_TIME = 45
# QUALITY_THRESHOLD = 0.6

# # -----------------------------
# # UI Configuration
# # -----------------------------
# GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
# GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
# GRADIO_QUEUE_SIZE = 10
# GRADIO_MAX_THREADS = 4
# GRADIO_ENABLE_STREAMING = True

# # -----------------------------
# # Logging Configuration
# # -----------------------------
# LOGGING_LEVEL = logging.INFO
# LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# # Configure logging
# def setup_logging():
#     """Setup logging configuration."""
#     # Create logs directory
#     logs_dir = STORAGE_ROOT / "logs"
#     logs_dir.mkdir(exist_ok=True)
    
#     # Configure logging
#     logging.basicConfig(
#         level=LOGGING_LEVEL,
#         format=LOGGING_FORMAT,
#         handlers=[
#             logging.StreamHandler(),
#             logging.FileHandler(logs_dir / "legal_chatbot.log", encoding="utf-8")
#         ]
#     )

# # Setup logging when module is imported
# setup_logging()

# # -----------------------------
# # Helper Functions
# # -----------------------------
# def get_logger(name: str) -> logging.Logger:
#     """Get a configured logger instance."""
#     return logging.getLogger(name)

# def validate_config() -> None:
#     """Validate configuration and raise errors for missing required settings."""
#     if not GEMINI_API_KEY and not OPENAI_API_KEY:
#         raise ValueError(
#             "No API key found. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file"
#         )

#     if LLM_PROVIDER not in ["gemini", "openai"]:
#         raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be 'gemini' or 'openai'")

#     if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
#         raise ValueError("GEMINI_API_KEY required when LLM_PROVIDER=gemini")

#     if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
#         raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")

#     # Validate token limits
#     if LLM_MAX_TOKENS < 500:
#         raise ValueError("LLM_MAX_TOKENS should be at least 500 for reasonable responses")

#     # Validate paths
#     if not PROJECT_ROOT.exists():
#         raise ValueError(f"Project root directory not found: {PROJECT_ROOT}")

# # Validate configuration on import
# try:
#     validate_config()
#     logger = get_logger(__name__)
#     logger.info(f"Legal Document AI Assistant configuration loaded")
#     logger.info(f"LLM Provider: {LLM_PROVIDER}")
#     logger.info(f"Max Tokens: {LLM_MAX_TOKENS}")
#     logger.info(f"Professional Mode: {PROFESSIONAL_MODE}")
# except ValueError as e:
#     # Don't fail import, just warn
#     logger = get_logger(__name__)
#     logger.warning(f"Configuration validation warning: {e}")

##########################################

"""
Fixed conversation session manager for the legal document chatbot.
Maintains conversation history and context with proper error handling.
"""

from typing import Dict, List, Any, Optional
import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict

# Import with fallback handling
try:
    from src.config import get_logger, MAX_CONVERSATION_HISTORY, MAX_CONTEXT_LENGTH
except ImportError:
    try:
        from .config import get_logger, MAX_CONVERSATION_HISTORY, MAX_CONTEXT_LENGTH
    except ImportError:
        import src.config as config
        get_logger = config.get_logger
        MAX_CONVERSATION_HISTORY = getattr(config, 'MAX_CONVERSATION_HISTORY', 10)
        MAX_CONTEXT_LENGTH = getattr(config, 'MAX_CONTEXT_LENGTH', 6000)

# Import agent with fallback
try:
    from src.agent import get_agent, answer_query
except ImportError:
    try:
        from .agent import get_agent, answer_query
    except ImportError:
        # Create fallback functions
        def get_agent():
            return MockAgent()
        
        def answer_query(query: str, context: Optional[Dict] = None):
            return {
                'answer': f"I understand you're asking about: {query}. However, I'm currently in fallback mode and cannot provide detailed analysis. Please ensure all components are properly configured.",
                'sources': [],
                'kg_nodes': [],
                'query_type': 'fallback',
                'confidence': 0.5,
                'processing_time': 1.0
            }

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    timestamp: str
    user_message: str
    bot_response: str
    sources: List[Dict[str, Any]]
    query_type: str
    confidence: float
    processing_time: float


class MockAgent:
    """Mock agent for fallback mode."""
    
    def answer_query(self, query: str, context: Optional[Dict] = None):
        """Provide mock response."""
        return {
            'answer': f"Based on legal documents and arbitration precedents, regarding your question about {query}: This is a mock response while the system is being configured. Please ensure all components are properly set up.",
            'sources': [],
            'kg_nodes': [],
            'query_type': 'mock',
            'confidence': 0.5,
            'processing_time': 1.0
        }


class ConversationSession:
    """Manages a single conversation session."""
    
    def __init__(self, user_id: str):
        """Initialize conversation session."""
        self.user_id = user_id
        self.session_id = f"{user_id}_{int(time.time())}"
        self.history: List[ConversationTurn] = []
        self.created_at = datetime.now().isoformat()
        self.last_activity = self.created_at
        
        logger.info(f"Created new conversation session: {self.session_id}")
    
    def add_turn(self, user_message: str, bot_response: Dict[str, Any]) -> None:
        """Add a conversation turn to history."""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_message=user_message,
            bot_response=bot_response.get('answer', ''),
            sources=bot_response.get('sources', []),
            query_type=bot_response.get('query_type', 'unknown'),
            confidence=bot_response.get('confidence', 0.0),
            processing_time=bot_response.get('processing_time', 0.0)
        )
        
        self.history.append(turn)
        self.last_activity = turn.timestamp
        
        # Maintain history limit
        if len(self.history) > MAX_CONVERSATION_HISTORY:
            self.history = self.history[-MAX_CONVERSATION_HISTORY:]
        
        logger.debug(f"Added turn to session {self.session_id}. History length: {len(self.history)}")
    
    def get_context(self) -> str:
        """
        Get conversation context for the LLM.
        
        Returns:
            Formatted conversation history
        """
        if not self.history:
            return ""
        
        context_parts = []
        total_length = 0
        
        # Add recent turns first (reverse chronological)
        for turn in reversed(self.history[-5:]):  # Last 5 turns
            turn_text = f"User: {turn.user_message}\nAssistant: {turn.bot_response[:200]}..."
            
            if total_length + len(turn_text) > MAX_CONTEXT_LENGTH:
                break
            
            context_parts.insert(0, turn_text)
            total_length += len(turn_text)
        
        return "\n\n".join(context_parts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        if not self.history:
            return {
                'session_id': self.session_id,
                'user_id': self.user_id,
                'total_turns': 0,
                'created_at': self.created_at,
                'last_activity': self.last_activity
            }
        
        # Calculate statistics
        query_types = [turn.query_type for turn in self.history]
        avg_confidence = sum(turn.confidence for turn in self.history) / len(self.history)
        avg_processing_time = sum(turn.processing_time for turn in self.history) / len(self.history)
        
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'total_turns': len(self.history),
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'query_types': list(set(query_types)),
            'avg_confidence': round(avg_confidence, 2),
            'avg_processing_time': round(avg_processing_time, 2)
        }
    
    def export_history(self) -> List[Dict[str, Any]]:
        """Export conversation history as JSON-serializable data."""
        return [asdict(turn) for turn in self.history]


class ChatbotManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self):
        """Initialize chatbot manager."""
        self.sessions: Dict[str, ConversationSession] = {}
        try:
            self.agent = get_agent()
            logger.info("Chatbot manager initialized with agent")
        except Exception as e:
            logger.warning(f"Agent initialization failed: {e}")
            self.agent = MockAgent()
            logger.info("Chatbot manager initialized with mock agent")
    
    def get_or_create_session(self, user_id: str) -> ConversationSession:
        """Get existing session or create new one."""
        if user_id not in self.sessions:
            self.sessions[user_id] = ConversationSession(user_id)
        
        return self.sessions[user_id]
    
    def handle_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Handle incoming user message.
        
        Args:
            user_id: Unique identifier for the user
            message: User's message
            
        Returns:
            Bot response with metadata
        """
        try:
            logger.info(f"Processing message from user {user_id}: {message[:100]}...")
            
            # Get or create session
            session = self.get_or_create_session(user_id)
            
            # Prepare context from conversation history
            conversation_context = {
                'history': session.get_context(),
                'session_id': session.session_id,
                'turn_number': len(session.history) + 1
            }
            
            # Get response from agent
            try:
                if hasattr(self.agent, 'provide_legal_counsel'):
                    # New agent interface
                    agent_response_obj = self.agent.provide_legal_counsel(message, conversation_context)
                    agent_response = {
                        'answer': agent_response_obj.answer,
                        'sources': agent_response_obj.sources,
                        'kg_nodes': agent_response_obj.kg_nodes,
                        'query_type': agent_response_obj.query_type,
                        'confidence': agent_response_obj.confidence,
                        'processing_time': agent_response_obj.processing_time
                    }
                else:
                    # Fallback to function-based interface
                    agent_response = answer_query(message, conversation_context)
            except Exception as e:
                logger.error(f"Agent error: {e}")
                agent_response = {
                    'answer': f"I apologize, but I encountered a technical issue while processing your legal question. Please try rephrasing your question or contact support if this continues.",
                    'sources': [],
                    'kg_nodes': [],
                    'query_type': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0
                }
            
            # Add to session history
            session.add_turn(message, agent_response)
            
            # Prepare response for UI
            response = {
                'answer': agent_response['answer'],
                'sources': agent_response['sources'],
                'kg_nodes': agent_response['kg_nodes'],
                'query_type': agent_response['query_type'],
                'confidence': agent_response['confidence'],
                'processing_time': agent_response['processing_time'],
                'session_info': {
                    'session_id': session.session_id,
                    'turn_number': len(session.history),
                    'total_turns': len(session.history)
                }
            }
            
            logger.info(f"Message processed successfully for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error handling message for user {user_id}: {e}")
            
            return {
                'answer': "I apologize, but I encountered an error processing your message. Please try again or rephrase your question.",
                'sources': [],
                'kg_nodes': [],
                'query_type': 'error',
                'confidence': 0.0,
                'processing_time': 0.0,
                'session_info': {
                    'error': str(e)
                }
            }
    
    def get_session_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of user's session."""
        if user_id in self.sessions:
            return self.sessions[user_id].get_summary()
        return None
    
    def export_session(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Export user's conversation history."""
        if user_id in self.sessions:
            return self.sessions[user_id].export_history()
        return None
    
    def clear_session(self, user_id: str) -> bool:
        """Clear user's conversation session."""
        if user_id in self.sessions:
            del self.sessions[user_id]
            logger.info(f"Cleared session for user {user_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session user IDs."""
        return list(self.sessions.keys())


# Global chatbot manager instance
_chatbot_manager = None

def get_chatbot_manager() -> ChatbotManager:
    """Get the global chatbot manager instance."""
    global _chatbot_manager
    if _chatbot_manager is None:
        _chatbot_manager = ChatbotManager()
    return _chatbot_manager


def handle_message(user_id: str, message: str) -> Dict[str, Any]:
    """
    Convenience function to handle messages.
    
    Args:
        user_id: User identifier
        message: User message
        
    Returns:
        Bot response
    """
    manager = get_chatbot_manager()
    return manager.handle_message(user_id, message)


def get_session_info(user_id: str) -> Optional[Dict[str, Any]]:
    """Get session information for user."""
    manager = get_chatbot_manager()
    return manager.get_session_summary(user_id)


def clear_user_session(user_id: str) -> bool:
    """Clear user's session."""
    manager = get_chatbot_manager()
    return manager.clear_session(user_id)


# Test function for development
def test_chatbot():
    """Test the chatbot functionality."""
    test_user = "test_user_1"
    
    # Test messages
    test_messages = [
        "What is the timeline for arbitration proceedings?",
        "How much do arbitration costs typically range?",
        "What makes an arbitration clause valid?"
    ]
    
    print("🧪 Testing chatbot functionality...")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {message}")
        
        response = handle_message(test_user, message)
        print(f"Bot: {response['answer'][:200]}...")
        print(f"Sources: {len(response['sources'])}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Processing time: {response['processing_time']:.2f}s")
    
    # Get session summary
    summary = get_session_info(test_user)
    print(f"\nSession Summary: {summary}")


if __name__ == "__main__":
    test_chatbot()