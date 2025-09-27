"""
Professional Prompt Engineering Module for Case Study Chatbot.
Provides sophisticated prompts for professional, business-focused responses.
"""

from typing import Dict, Any, List
from datetime import datetime
class ResponseFormatter:
    """Formats responses for arbitration and business analysis using attached data."""
    
    def __init__(self):
        """Initialize with simple formatting templates."""
        self.section_headers = {
            'summary': '## Summary',
            'details': '## Details',
            'sources': '## Sources',
            'next_steps': '## Next Steps'
        }

    def format_professional_response(self, response: dict) -> str:
        """
        Format response in a simple, professional style.

        Args:
            response: Dictionary with 'answer', 'sources', 'query_type', and optional 'confidence'.

        Returns:
            Formatted response string.
        """
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        query_type = response.get('query_type', 'general')

        # Start with main answer
        formatted = f"{self.section_headers['summary']}\n\n{answer}\n\n"

        # Add details based on query type
        formatted += self._add_details_section(answer, query_type)

        # Add sources if available
        if sources:
            formatted += self._add_source_section(sources)

        # Add next steps
        formatted += self._add_next_steps(query_type)

        return formatted

    def _add_details_section(self, answer: str, query_type: str) -> str:
        """Add a details section based on query type."""
        details = f"{self.section_headers['details']}\n\n"

        if query_type == 'analysis':
            details += "- Analysis based on decision trees or case studies from attached documents.\n"
            details += "- Key points derived from relevant arbitration cases or business examples.\n"
        elif query_type == 'recommendation':
            details += "- Recommendations drawn from documents like 'Cost of arbitration and compared to claimed.docx'.\n"
            details += "- Supported by case studies, e.g., QualityTech in 'advanced_cases.docx'.\n"
        elif query_type == 'comparison':
            details += "- Comparison using data from arbitration cases in 'NLP22222.docx'.\n"
            details += "- Highlights differences and outcomes.\n"
        elif query_type == 'trend':
            details += "- Trends from past and present cases in 'NLP22222.docx'.\n"
            details += "- Future insights from decision trees like 'Theory of emergency conditions.docx'.\n"
        elif query_type == 'implementation':
            details += "- Steps based on timelines in 'Time framed for arbitration process (guide).docx'.\n"
            details += "- Actions from decision trees, e.g., 'Validity of arbitration clause 21082025.docx'.\n"
        else:
            details += "- Response based on attached arbitration and business data.\n"

        return details

    def _add_source_section(self, sources: list) -> str:
        """Add a simple source section."""
        source_section = f"{self.section_headers['sources']}\n\n"
        for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources
            source_id = source.get('id', f'Source_{i}')
            preview = source.get('text', '')[:150] + '...' if len(source.get('text', '')) > 150 else source.get('text', '')
            source_section += f"- **{source_id}**: {preview}\n"
        return source_section + "\n"

    def _add_next_steps(self, query_type: str) -> str:
        """Add simple next steps based on query type."""
        next_steps = f"{self.section_headers['next_steps']}\n\n"
        if query_type == 'analysis':
            next_steps += "- Review decision trees (e.g., 'Cassation_of_Arbitral_Award_Restructured.docx') for next actions.\n"
            next_steps += "- Check arbitration cases in 'NLP22222.docx' for precedents.\n"
        elif query_type == 'recommendation':
            next_steps += "- Implement recommended actions using case study examples.\n"
            next_steps += "- Refer to 'advanced_cases.docx' for practical guidance.\n"
        elif query_type == 'comparison':
            next_steps += "- Choose the best option based on comparison results.\n"
            next_steps += "- Validate with additional cases from 'NLP22222.docx'.\n"
        elif query_type == 'trend':
            next_steps += "- Monitor trends using recent cases in 'NLP22222.docx'.\n"
            next_steps += "- Plan for future scenarios using decision trees.\n"
        elif query_type == 'implementation':
            next_steps += "- Follow timeline in 'Time framed for arbitration process (guide).docx'.\n"
            next_steps += "- Use decision trees for procedural steps.\n"
        else:
            next_steps += "- Ask follow-up questions to deepen analysis.\n"
            next_steps += "- Consult attached documents for more details.\n"
        return next_steps


class ConversationFormatter:
    """Formats conversation history and session summaries."""
    
    def __init__(self):
        """Initialize conversation formatter."""
        pass
    
    def format_chat_history(self, history: list, max_display: int = 5) -> str:
        """Format recent conversation history."""
        if not history:
            return "## Conversation History\n\nNo previous conversations.\n"

        formatted = "## Conversation History\n\n"
        for i, (user_msg, bot_response) in enumerate(history[-max_display:], 1):
            formatted += f"**Exchange {i}**\n"
            formatted += f"- **User**: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}\n"
            formatted += f"- **Assistant**: {bot_response[:100]}{'...' if len(bot_response) > 100 else ''}\n\n"
        return formatted

    def format_session_summary(self, session_info: dict) -> str:
        """Format a simple session summary."""
        total_turns = session_info.get('total_turns', 0)
        query_types = session_info.get('query_types', [])
        
        summary = "## Session Summary\n\n"
        summary += f"- **Total Interactions**: {total_turns}\n"
        summary += f"- **Query Types**: {', '.join(q.title().replace('_', ' ') for q in query_types) or 'None'}\n"
        summary += "- **Next Steps**: Continue with specific questions or explore attached documents.\n"
        return summary


# Global formatter instances
_response_formatter = None
_conversation_formatter = None

def get_response_formatter() -> ResponseFormatter:
    """Get the global response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatter()
    return _response_formatter

def get_conversation_formatter() -> ConversationFormatter:
    """Get the global conversation formatter instance."""
    global _conversation_formatter
    if _conversation_formatter is None:
        _conversation_formatter = ConversationFormatter()
    return _conversation_formatter

def format_response(response: dict, style: str = 'professional') -> str:
    """
    Format response based on style.

    Args:
        response: Dictionary with response data.
        style: Formatting style ('professional' only).

    Returns:
        Formatted response string.
    """
    formatter = get_response_formatter()
    return formatter.format_professional_response(response)

def format_conversation_history(history: list, max_display: int = 5) -> str:
    """Format conversation history."""
    formatter = get_conversation_formatter()
    return formatter.format_chat_history(history, max_display)

def format_session_summary(session_info: dict) -> str:
    """Format session summary."""
    formatter = get_conversation_formatter()
    return formatter.format_session_summary(session_info)

# Export public functions
__all__ = [
    'ResponseFormatter',
    'ConversationFormatter',
    'get_response_formatter',
    'get_conversation_formatter',
    'format_response',
    'format_conversation_history',
    'format_session_summary'
]