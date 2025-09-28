
"""
Simple response formatter for humanized legal counsel responses.
"""

from typing import Dict, Any, List


class ResponseFormatter:
    """Simple formatter for humanized legal counsel responses."""
    
    def __init__(self):
        """Initialize with personalized legal formatting."""
        self.section_headers = {
            'summary': '## 📋 Legal Summary',
            'analysis': '## ⚖️ Legal Analysis', 
            'sources': '## 📚 Referenced Documents',
            'advice': '## 💼 My Legal Advice'
        }

    def format_professional_response(self, response: Dict[str, Any]) -> str:
        """
        Format response as a humanized legal professional.

        Args:
            response: Dictionary with 'answer', 'sources', 'query_type', etc.

        Returns:
            Humanized, concise legal response.
        """
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        query_type = response.get('query_type', 'general')
        confidence = response.get('confidence', 0.0)

        # Start with the legal counsel's answer (already humanized)
        formatted = f"{answer}\n\n"

        # Add concise legal context
        formatted += self._add_legal_context(query_type, confidence)

        # Add source references if available
        if sources:
            formatted += self._add_legal_sources(sources)

        # Add professional footer
        formatted += self._add_legal_footer(confidence)

        return formatted

    def _add_legal_context(self, query_type: str, confidence: float) -> str:
        """Add brief legal context."""
        context = "---\n"
        
        # Confidence assessment in legal terms
        if confidence >= 0.8:
            context += "**Legal Assessment:** High confidence - Strong documentary support\n"
        elif confidence >= 0.6:
            context += "**Legal Assessment:** Good confidence - Adequate legal basis\n"
        else:
            context += "**Legal Assessment:** Preliminary view - Additional research recommended\n"
        
        # Query type in legal terms
        if query_type == 'direct_legal_advice':
            context += "**Response Type:** Direct legal guidance\n"
        elif query_type == 'analysis':
            context += "**Response Type:** Legal analysis\n"
        elif query_type == 'recommendation':
            context += "**Response Type:** Legal recommendations\n"
        else:
            context += "**Response Type:** Legal consultation\n"
        
        return context + "\n"

    def _add_legal_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Add concise source references."""
        if not sources:
            return ""
            
        source_section = f"{self.section_headers['sources']}\n"
        
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            source_id = source.get('id', f'Document_{i}')
            relevance = source.get('relevance', 'Supporting')
            text_preview = source.get('text', '')
            
            # Smart preview for legal content
            if len(text_preview) > 100:
                preview = text_preview[:100] + "..."
            else:
                preview = text_preview
                
            source_section += f"📄 **{source_id}** ({relevance} relevance)\n"
            source_section += f"   *{preview}*\n\n"
        
        return source_section

    def _add_legal_footer(self, confidence: float) -> str:
        """Add professional legal footer."""
        footer = "---\n"
        footer += "**💼 Legal Note:** "
        
        if confidence >= 0.7:
            footer += "This analysis is based on available legal documentation and established precedents. "
        else:
            footer += "This preliminary assessment requires additional documentation for complete analysis. "
        
        footer += "For formal legal proceedings, please ensure all documentation is current and complete.\n"
        
        return footer


class ConversationFormatter:
    """Simple conversation history formatter for legal consultations."""
    
    def format_chat_history(self, history: List[List[str]], max_display: int = 5) -> str:
        """Format recent legal consultation history."""
        if not history:
            return "## 💬 Consultation History\nNo previous consultations in this session.\n"

        formatted = "## 💬 Recent Legal Consultations\n\n"
        recent_history = history[-max_display:] if len(history) > max_display else history
        
        for i, (user_msg, bot_response) in enumerate(recent_history, 1):
            # Get just the main legal advice (before metadata)
            main_advice = bot_response.split('\n---\n')[0] if '\n---\n' in bot_response else bot_response
            
            formatted += f"**Consultation {i}**\n"
            formatted += f"🤝 **Client:** {user_msg[:120]}{'...' if len(user_msg) > 120 else ''}\n"
            formatted += f"⚖️ **Counsel:** {main_advice[:150]}{'...' if len(main_advice) > 150 else ''}\n\n"
        
        return formatted

    def format_session_summary(self, session_info: Dict[str, Any]) -> str:
        """Format legal consultation session summary."""
        if not session_info:
            return "## 📊 Session Summary\nNew consultation - ready for your legal questions!\n"
            
        total_turns = session_info.get('total_turns', 0)
        avg_confidence = session_info.get('avg_confidence', 0)
        
        summary = "## 📊 Legal Consultation Summary\n\n"
        summary += f"**Questions Addressed:** {total_turns}\n"
        
        if avg_confidence > 0:
            confidence_assessment = "Strong" if avg_confidence >= 0.7 else "Good" if avg_confidence >= 0.5 else "Preliminary"
            summary += f"**Analysis Quality:** {confidence_assessment}\n"
        
        summary += f"**Focus Area:** Arbitration & Commercial Law\n"
        summary += f"**Legal Counsel:** Available for follow-up questions\n"
        
        return summary


# Global instances
_response_formatter = None
_conversation_formatter = None

def get_response_formatter() -> ResponseFormatter:
    """Get the response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatter()
    return _response_formatter

def get_conversation_formatter() -> ConversationFormatter:
    """Get the conversation formatter instance."""
    global _conversation_formatter
    if _conversation_formatter is None:
        _conversation_formatter = ConversationFormatter()
    return _conversation_formatter

def format_response(response: Dict[str, Any], style: str = 'professional') -> str:
    """Format response simply and professionally."""
    formatter = get_response_formatter()
    return formatter.format_professional_response(response)

def format_conversation_history(history: List[List[str]], max_display: int = 5) -> str:
    """Format conversation history."""
    formatter = get_conversation_formatter()
    return formatter.format_chat_history(history, max_display)

def format_session_summary(session_info: Dict[str, Any]) -> str:
    """Format session summary."""
    formatter = get_conversation_formatter()
    return formatter.format_session_summary(session_info)

# Export functions
__all__ = [
    'ResponseFormatter',
    'ConversationFormatter', 
    'get_response_formatter',
    'get_conversation_formatter',
    'format_response',
    'format_conversation_history',
    'format_session_summary'
]
