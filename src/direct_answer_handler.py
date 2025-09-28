
"""
Simple direct answer handler for arbitration and legal questions.
Focused on providing clear, direct answers from legal documents.
"""

import re
from typing import Dict, Any, Optional, List
from src.config import get_logger

logger = get_logger(__name__)


class DirectAnswerHandler:
    """Simple handler for direct legal and arbitration questions."""
    
    def __init__(self):
        """Initialize with patterns for legal/arbitration questions."""
        
        # Simple patterns for direct questions
        self.direct_question_patterns = [
            r'^what is\s+',
            r'^what are\s+',
            r'^how long\s+',
            r'^how much\s+',
            r'^when\s+',
            r'^where\s+',
            r'^who\s+',
            r'^which\s+',
            r'^define\s+',
            r'^explain\s+',
        ]
        
        # Legal/arbitration specific keywords
        self.legal_keywords = [
            'arbitration', 'clause', 'timeline', 'cost', 'procedure', 
            'emergency', 'decision', 'award', 'validity', 'process',
            'deadline', 'filing', 'hearing', 'tribunal', 'mediation'
        ]
    
    def is_direct_question(self, query: str) -> bool:
        """Check if the query is a direct question about legal/arbitration topics."""
        query_lower = query.lower().strip()
        
        # Check direct question patterns
        for pattern in self.direct_question_patterns:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        
        # Check for question mark with legal keywords
        if query.endswith('?'):
            if any(keyword in query_lower for keyword in self.legal_keywords):
                return True
            if len(query.split()) <= 8:  # Short questions
                return True
        
        return False
    
    def analyze_context_for_answer(self, query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if we can provide a direct answer from legal documents."""
        sources = context_data.get('sources', [])
        
        if not sources:
            return {
                'can_answer': False,
                'confidence': 0.0,
                'answer_type': 'no_context',
                'reason': 'No relevant legal documents found'
            }
        
        # Calculate relevance for legal content
        legal_relevance = 0
        for source in sources:
            text = source.get('text', '').lower()
            score = source.get('score', 0)
            
            # Boost score if legal keywords found
            legal_matches = sum(1 for keyword in self.legal_keywords if keyword in text)
            if legal_matches > 0:
                legal_relevance += score + (legal_matches * 0.1)
        
        avg_relevance = legal_relevance / len(sources) if sources else 0
        
        # Determine answer confidence
        confidence = min(avg_relevance, 1.0)
        can_answer = confidence >= 0.4 and len(sources) > 0
        
        return {
            'can_answer': can_answer,
            'confidence': confidence,
            'answer_type': 'legal_direct',
            'legal_relevance': legal_relevance,
            'reason': 'Sufficient legal document context' if can_answer else 'Limited legal context'
        }
    
    def extract_key_facts(self, sources: List[Dict[str, Any]], query: str) -> List[str]:
        """Extract key facts from legal documents relevant to the query."""
        query_words = set(query.lower().split())
        facts = []
        
        for source in sources[:3]:  # Use top 3 sources
            text = source.get('text', '')
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                sentence_words = set(sentence.lower().split())
                
                # Check relevance to query
                overlap = len(query_words.intersection(sentence_words))
                has_legal_terms = any(term in sentence.lower() for term in self.legal_keywords)
                
                if overlap >= 1 or has_legal_terms:
                    facts.append(sentence)
        
        # Remove duplicates and return top facts
        unique_facts = list(dict.fromkeys(facts))
        return unique_facts[:3]  # Return top 3 facts
    
    def generate_direct_answer(self, query: str, context_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate a simple, direct answer from legal documents."""
        sources = context_data.get('sources', [])
        
        if not sources:
            return "I don't have enough information in the legal documents to answer that question."
        
        # Extract key facts
        key_facts = self.extract_key_facts(sources, query)
        
        if not key_facts:
            return "While I found some relevant legal documents, I couldn't extract specific information to answer your question directly."
        
        # Generate simple direct answer
        if len(key_facts) == 1:
            return f"Based on the legal documents: {key_facts[0]}"
        else:
            # Combine multiple facts
            answer = "Based on the legal documents:\n\n"
            for i, fact in enumerate(key_facts, 1):
                answer += f"{i}. {fact}\n"
            return answer.strip()


# Global instance
_direct_answer_handler = None

def get_direct_answer_handler() -> DirectAnswerHandler:
    """Get the global direct answer handler instance."""
    global _direct_answer_handler
    if _direct_answer_handler is None:
        _direct_answer_handler = DirectAnswerHandler()
    return _direct_answer_handler


def handle_direct_question(query: str, context_data: Dict[str, Any]) -> Optional[str]:
    """
    Handle direct legal/arbitration questions.
    
    Args:
        query: User query
        context_data: Retrieved context from legal documents
        
    Returns:
        Direct answer string or None if can't answer directly
    """
    handler = get_direct_answer_handler()
    
    # Analyze if we can provide a direct answer
    analysis = handler.analyze_context_for_answer(query, context_data)
    
    if not analysis['can_answer']:
        logger.info(f"Cannot provide direct answer: {analysis['reason']}")
        return None
    
    # Generate direct answer
    direct_answer = handler.generate_direct_answer(query, context_data, analysis)
    logger.info(f"Generated direct answer with confidence: {analysis['confidence']:.2f}")
    
    return direct_answer


def is_direct_question(query: str) -> bool:
    """Check if query is a direct question about legal/arbitration topics."""
    handler = get_direct_answer_handler()
    return handler.is_direct_question(query)
