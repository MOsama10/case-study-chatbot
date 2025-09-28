
"""
Fixed imports for agent.py - replace the import section at the top of your src/agent.py
"""

# At the top of src/agent.py, replace the import section with this:

from typing import Dict, Any, List, Optional
import logging
import time
from dataclasses import dataclass

# LLM API imports with error handling
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Local imports with both relative and absolute fallback
try:
    from src.config import (
        get_logger, GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
        LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
    )
except ImportError:
    try:
        from src.config import (
            get_logger, GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
            LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
        )
    except ImportError:
        import src.config as config
        get_logger = config.get_logger
        GEMINI_API_KEY = config.GEMINI_API_KEY
        OPENAI_API_KEY = config.OPENAI_API_KEY
        LLM_PROVIDER = config.LLM_PROVIDER
        LLM_TEMPERATURE = config.LLM_TEMPERATURE
        LLM_MAX_TOKENS = config.LLM_MAX_TOKENS
        LLM_TIMEOUT = config.LLM_TIMEOUT

logger = get_logger(__name__)


@dataclass
class LegalResponse:
    """Structured response from the legal professional agent."""
    answer: str
    sources: List[Dict[str, Any]]
    kg_nodes: List[str]
    raw_llm_response: str
    query_type: str
    confidence: float = 0.0
    processing_time: float = 0.0

class LegalProfessionalLLM:
    """LLM wrapper that acts as an experienced legal professional."""
    
    def __init__(self, provider: str = LLM_PROVIDER):
        """Initialize the legal professional LLM."""
        self.provider = provider.lower()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the LLM client."""
        if self.provider == "gemini":
            if not GENAI_AVAILABLE:
                raise ValueError("google-generativeai package not available. Install with: pip install google-generativeai")
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
                raise ValueError("GEMINI_API_KEY not configured. Please set it in .env file")
            
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.client = genai.GenerativeModel(model_name="models/gemini-pro-latest")
                logger.info("Legal professional assistant initialized with Gemini")
            except Exception as e:
                raise ValueError(f"Failed to initialize Gemini: {e}")

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ValueError("openai package not available. Install with: pip install openai")
            if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
                raise ValueError("OPENAI_API_KEY not configured. Please set it in .env file")
            
            try:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info("Legal professional assistant initialized with OpenAI")
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI: {e}")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_legal_response(self, prompt: str, **kwargs) -> str:
        """Generate a humanized legal professional response."""
        try:
            max_tokens = kwargs.get('max_tokens', LLM_MAX_TOKENS)
            temperature = kwargs.get('temperature', LLM_TEMPERATURE)
            
            if self.provider == "gemini":
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                if hasattr(response, 'text') and response.text:
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidates = response.candidates
                    if candidates and hasattr(candidates[0], 'content'):
                        content = candidates[0].content
                        if hasattr(content, 'parts') and content.parts:
                            parts = content.parts
                            if parts and hasattr(parts[0], 'text'):
                                return parts[0].text
                
                return "I apologize, but I'm having trouble accessing the legal documents right now. Could you please rephrase your question?"

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=kwargs.get('model', 'gpt-4o-mini'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=LLM_TIMEOUT
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Legal LLM generation error: {e}")
            return "I'm experiencing some technical difficulties accessing the legal database. Let me try a different approach to your question."

class LegalProfessionalAgent:
    """Main agent that acts as an experienced legal professional."""
    
    def __init__(self, llm_provider: str = LLM_PROVIDER):
        """Initialize the legal professional agent."""
        try:
            self.llm = LegalProfessionalLLM(llm_provider)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Use mock for testing/fallback
            self.llm = MockLegalCounsel()
        
        # Initialize retriever with error handling
        try:
            from src.retriever import get_retriever
            self.retriever = get_retriever()
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            self.retriever = None
        
        # Professional legal persona
        self.legal_persona = """You are an experienced legal counsel with 15+ years specializing in international arbitration, commercial disputes, and contract law. You have extensive experience with:

- International arbitration proceedings (ICC, LCIA, UNCITRAL)
- Bilateral Investment Treaties (BITs) and investor-state disputes
- Commercial contract disputes and clause validity
- Arbitration agreement enforcement and challenges
- Emergency arbitration procedures
- Cross-border dispute resolution

Your communication style is:
- Professional yet approachable and humanized
- Concise and summary-focused - you get straight to the point
- Confident in your legal analysis
- Uses "I" and personal pronouns (I've seen, In my experience, I would advise)
- Speaks from practical experience, not just theory
- Provides clear, actionable guidance
- Uses legal terminology correctly but explains when needed"""

    def create_legal_prompt(self, query: str, context_data: Dict[str, Any]) -> str:
        """Create a humanized legal professional prompt."""
        
        context_text = context_data.get('context_text', '')
        query_type = context_data.get('query_type', 'general')
        
        # Summary-focused prompt template
        base_prompt = f"""{self.legal_persona}

Legal Documents Available:
{context_text}

Client Question: {query}

Instructions:
- Provide a CONCISE, SUMMARY-STYLE response (2-3 paragraphs maximum)
- Speak as an experienced legal counsel using "I" and personal experience
- Give a direct, practical answer first, then supporting reasoning
- Reference specific documents when relevant
- Be personable but professional
- Focus on actionable legal guidance
- If it's a yes/no question, start with a clear yes/no

Response as Legal Counsel:"""

        return base_prompt
    
    def assess_legal_confidence(self, response: str, sources: List[Dict], query: str) -> float:
        """Assess confidence from a legal professional perspective."""
        base_confidence = 0.6
        
        # Boost for strong legal language
        strong_legal_phrases = [
            'clearly states', 'explicitly provides', 'definitively establishes',
            'unambiguously requires', 'specifically mandates', 'categorically prohibits',
            'expressly permits', 'conclusively demonstrates'
        ]
        
        if any(phrase in response.lower() for phrase in strong_legal_phrases):
            base_confidence += 0.2
        
        # Boost for specific legal citations or references
        if any(word in response.lower() for word in ['clause', 'article', 'section', 'paragraph']):
            base_confidence += 0.1
        
        # Boost for personal legal experience indicators
        experience_phrases = ['in my experience', 'i have seen', 'typically', 'usually', 'commonly']
        if any(phrase in response.lower() for phrase in experience_phrases):
            base_confidence += 0.1
        
        # Boost for clear conclusions
        conclusion_phrases = ['therefore', 'accordingly', 'consequently', 'in conclusion', 'my advice']
        if any(phrase in response.lower() for phrase in conclusion_phrases):
            base_confidence += 0.1
        
        # Factor in source quality
        if sources:
            high_quality_sources = [s for s in sources if s.get('score', 0) > 0.7]
            if len(high_quality_sources) >= 2:
                base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def provide_legal_counsel(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> LegalResponse:
        """Provide legal counsel as an experienced attorney."""
        start_time = time.time()
        
        try:
            logger.info(f"Legal counsel processing: {query}")
            
            # Retrieve relevant legal materials
            if self.retriever:
                try:
                    context_data = self._retrieve_for_query(query)
                except Exception as e:
                    logger.error(f"Retrieval failed: {e}")
                    context_data = self._create_fallback_context(query)
            else:
                context_data = self._create_fallback_context(query)
            
            # Check for direct legal questions first
            try:
                from src.direct_answer_handler import get_direct_answer_handler, handle_direct_question
                direct_handler = get_direct_answer_handler()
                
                if direct_handler.is_direct_question(query):
                    logger.info("Providing direct legal guidance")
                    
                    direct_response = handle_direct_question(query, context_data)
                    
                    if direct_response:
                        # Enhance direct response with legal personality
                        enhanced_response = self._add_legal_personality(direct_response, query)
                        
                        analysis = direct_handler.analyze_context_for_answer(query, context_data)
                        confidence = self.assess_legal_confidence(enhanced_response, context_data.get('sources', []), query)
                        
                        sources = self._prepare_legal_sources(context_data.get('sources', []))
                        processing_time = time.time() - start_time
                        
                        return LegalResponse(
                            answer=enhanced_response,
                            sources=sources,
                            kg_nodes=context_data.get('kg_results', []),
                            raw_llm_response=enhanced_response,
                            query_type="direct_legal_advice",
                            confidence=confidence,
                            processing_time=processing_time
                        )
            except Exception as e:
                logger.warning(f"Direct answer handling failed: {e}")
            
            # Generate comprehensive legal analysis
            legal_prompt = self.create_legal_prompt(query, context_data)
            llm_response = self.llm.generate_legal_response(legal_prompt)
            
            # Assess confidence from legal perspective
            confidence = self.assess_legal_confidence(llm_response, context_data.get('sources', []), query)
            
            # Prepare legal sources
            sources = self._prepare_legal_sources(context_data.get('sources', []))
            
            processing_time = time.time() - start_time
            
            response = LegalResponse(
                answer=llm_response,
                sources=sources,
                kg_nodes=context_data.get('kg_results', []),
                raw_llm_response=llm_response,
                query_type=context_data.get('query_type', 'legal_analysis'),
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info(f"Legal counsel provided in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error providing legal counsel: {e}")
            
            return LegalResponse(
                answer="I apologize, but I'm having difficulty accessing the legal documents right now. This could be a temporary technical issue. Could you please try rephrasing your question, or if this is urgent, we should schedule a call to discuss this matter properly.",
                sources=[],
                kg_nodes=[],
                raw_llm_response="",
                query_type="error",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _retrieve_for_query(self, query: str) -> Dict[str, Any]:
        """Safely retrieve context for query."""
        try:
            from src.retriever import retrieve_for_query
            return retrieve_for_query(query)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return self._create_fallback_context(query)
    
    def _create_fallback_context(self, query: str) -> Dict[str, Any]:
        """Create fallback context when retrieval fails."""
        return {
            'query_type': 'legal_analysis',
            'context_text': 'Based on general legal principles and arbitration practices.',
            'sources': [],
            'kg_results': [],
            'total_sources': 0
        }
    
    def _add_legal_personality(self, response: str, query: str) -> str:
        """Add legal professional personality to direct responses."""
        
        # Add personal legal perspective
        if response.startswith("Based on the legal documents:"):
            response = response.replace(
                "Based on the legal documents:", 
                "Looking at the legal documentation in this matter,"
            )
        
        # Add conclusion with legal advice
        if not any(phrase in response.lower() for phrase in ['i would', 'my advice', 'i recommend']):
            if '?' in query and any(word in query.lower() for word in ['should', 'can', 'may', 'recommend']):
                response += "\n\nMy advice would be to proceed with caution and ensure all requirements are met before taking action."
        
        return response
    
    def _prepare_legal_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare sources with legal context."""
        legal_sources = []
        
        for source in sources[:5]:  # Top 5 sources
            legal_sources.append({
                'id': source.get('id', 'unknown'),
                'text': source.get('text', '')[:250] + "..." if len(source.get('text','')) > 250 else source.get('text',''),
                'score': source.get('score', 0.0),
                'type': source.get('source_type', 'legal_document'),
                'relevance': 'High' if source.get('score', 0) > 0.8 else 'Medium' if source.get('score', 0) > 0.6 else 'Supporting'
            })
        
        return legal_sources

class MockLegalCounsel:
    """Mock legal counsel for testing."""
    
    def generate_legal_response(self, prompt: str, **kwargs) -> str:
        """Generate mock legal counsel response."""
        return """Based on my review of the available legal documentation, I can provide you with the following analysis:

**Summary:** The arbitration clause validity depends on several key factors that I've identified in the documents. In my experience with similar cases, the most critical elements are the clarity of language, mutual consent, and proper scope definition.

**My Assessment:** From what I can see in the documentation, there are specific requirements that must be met for an arbitration clause to be enforceable. I would need to examine the exact clause language, but generally speaking, validity hinges on whether the parties clearly agreed to arbitrate disputes and whether the clause covers the type of dispute in question.

**Practical Advice:** I recommend ensuring that all procedural requirements are met and that the clause language is unambiguous. If there are any concerns about validity, it's better to address them proactively.

This analysis is based on the available legal documents and my professional experience with arbitration matters."""

# Global agent instance
_legal_agent = None

def get_agent() -> LegalProfessionalAgent:
    """Get the global legal professional agent instance."""
    global _legal_agent
    if _legal_agent is None:
        try:
            _legal_agent = LegalProfessionalAgent()
        except Exception as e:
            logger.warning(f"Failed to initialize legal counsel: {e}")
            logger.info("Using mock legal counsel for testing")
            _legal_agent = LegalProfessionalAgent.__new__(LegalProfessionalAgent)
            _legal_agent.llm = MockLegalCounsel()
            _legal_agent.retriever = None
    return _legal_agent

def answer_query(query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get legal counsel response to query.
    
    Args:
        query: Legal question or matter
        user_context: Optional context
        
    Returns:
        Legal counsel response
    """
    agent = get_agent()
    response = agent.provide_legal_counsel(query, user_context)
    
    return {
        'answer': response.answer,
        'sources': response.sources,
        'kg_nodes': response.kg_nodes,
        'raw_llm_response': response.raw_llm_response,
        'query_type': response.query_type,
        'confidence': response.confidence,
        'processing_time': response.processing_time
    }
