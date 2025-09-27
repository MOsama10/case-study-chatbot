"""
Agentic RAG implementation for case study analysis.
Orchestrates retrieval and LLM reasoning.
"""

from typing import Dict, Any, List, Optional
import logging
import time
import json
from dataclasses import dataclass

# LLM API imports
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

from .config import (
    get_logger, GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
)
from .retriever import get_retriever, retrieve_for_query

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Structured response from the agent."""
    answer: str
    sources: List[Dict[str, Any]]
    kg_nodes: List[str]
    raw_llm_response: str
    query_type: str
    confidence: float = 0.0
    processing_time: float = 0.0


class LLMWrapper:
    """Wrapper for different LLM providers."""
    
    def __init__(self, provider: str = LLM_PROVIDER):
        """Initialize LLM wrapper with specified provider."""
        self.provider = provider.lower()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate LLM client."""
        if self.provider == "gemini":
            if not GENAI_AVAILABLE or not GEMINI_API_KEY:
                raise ValueError("Gemini API not available or API key not set")
            
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)

            # If model name not specified in config, pick latest supported generateContent model
            try:
                from .config import GEMINI_MODEL
            except ImportError:
                GEMINI_MODEL = None

            if not GEMINI_MODEL:
                logger.info("No Gemini model set in config. Selecting latest supported model automatically...")
                available_models = genai.list_models()
                # Filter models that support generateContent
                valid_models = [m for m in available_models if 'generateContent' in m.supported_methods]
                if not valid_models:
                    raise ValueError("No Gemini models supporting generateContent found.")
                # Pick the latest one by name (assuming naming reflects versions)
                GEMINI_MODEL = sorted(valid_models, key=lambda m: m.name, reverse=True)[0].name
                logger.info(f"Selected Gemini model: {GEMINI_MODEL}")

            self.client = genai.GenerativeModel(GEMINI_MODEL)
            logger.info(f"Initialized Gemini LLM client with model: {GEMINI_MODEL}")

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
                raise ValueError("OpenAI API not available or API key not set")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("Initialized OpenAI LLM client")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            if self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get('temperature', LLM_TEMPERATURE),
                        max_output_tokens=kwargs.get('max_tokens', LLM_MAX_TOKENS),
                    )
                )

                # Safely extract text
                if hasattr(response, 'text') and response.text:
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    parts = getattr(response.candidates[0].content, 'parts', None)
                    if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
                        return parts[0].text
                # fallback if no valid content
                return "I apologize, but the LLM did not return a valid response."

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=kwargs.get('model', 'gpt-3.5-turbo'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', LLM_TEMPERATURE),
                    max_tokens=kwargs.get('max_tokens', LLM_MAX_TOKENS),
                    timeout=LLM_TIMEOUT
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"I apologize, but an error occurred during LLM generation: {str(e)}"



class CaseStudyAgent:
    """Main agent for case study analysis."""
    
    def __init__(self, llm_provider: str = LLM_PROVIDER):
        """Initialize the agent."""
        self.llm = LLMWrapper(llm_provider)
        self.retriever = get_retriever()
        
        # Prompt templates
        self.system_prompt = """You are an expert case study analyst. Your role is to analyze business case studies and provide insightful, evidence-based responses.

Key responsibilities:
1. Analyze the provided context from case studies
2. Answer questions based on evidence from the sources
3. Identify patterns and relationships between problems and solutions
4. Provide actionable insights and recommendations
5. Always cite your sources using the provided source IDs

Guidelines:
- Be analytical and objective
- Support your answers with specific evidence from the sources
- Acknowledge when information is insufficient
- Focus on actionable insights
- Use clear, professional language"""

        self.qa_template = """Context from case studies:
{context}

Question: {query}

Please provide a comprehensive answer based on the case study context above. Include:
1. Direct answer to the question
2. Supporting evidence from the sources (cite source IDs)
3. Key insights or patterns identified
4. Any limitations or areas needing more information

Answer:"""

        self.analysis_template = """Context from case studies:
{context}

Analysis request: {query}

Please provide a detailed analysis based on the case study context above. Include:
1. Overview of the situation/topic
2. Key problems or challenges identified
3. Solutions or approaches used
4. Results and outcomes
5. Patterns and insights
6. Recommendations for similar situations
7. Source citations for all claims

Analysis:"""

    def build_prompt(self, query: str, context_data: Dict[str, Any]) -> str:
        """
        Build appropriate prompt based on query type.
        
        Args:
            query: User query
            context_data: Retrieved context information
            
        Returns:
            Formatted prompt string
        """
        query_type = context_data['query_type']
        context_text = context_data['context_text']
        
        if query_type == 'analysis':
            template = self.analysis_template
        else:
            template = self.qa_template
        
        # Add source information for better citation
        sources_info = "\n".join([
            f"Source {i+1} (ID: {source['id']}): {source['text'][:200]}..."
            for i, source in enumerate(context_data['sources'][:5])
        ])
        
        enhanced_context = f"{context_text}\n\nSource Details:\n{sources_info}"
        
        return template.format(
            context=enhanced_context,
            query=query
        )
    
    def extract_confidence(self, response: str) -> float:
        """
        Extract confidence score from LLM response.
        Simple heuristic-based approach.
        
        Args:
            response: LLM response text
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristics for confidence
        confidence_indicators = {
            'certain': 0.9,
            'confident': 0.8,
            'likely': 0.7,
            'probably': 0.6,
            'possibly': 0.5,
            'uncertain': 0.3,
            'unclear': 0.2
        }
        
        response_lower = response.lower()
        
        # Check for confidence indicators
        max_confidence = 0.5  # Default
        for indicator, score in confidence_indicators.items():
            if indicator in response_lower:
                max_confidence = max(max_confidence, score)
        
        # Adjust based on response length and detail
        if len(response) > 500:
            max_confidence += 0.1
        
        # Check for source citations
        if any(word in response_lower for word in ['source', 'according to', 'based on']):
            max_confidence += 0.1
        
        return min(max_confidence, 1.0)
    def answer_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Main method to answer user queries.
        
        Args:
            query: User question or request
            user_context: Optional user context (conversation history, etc.)
            
        Returns:
            Structured AgentResponse
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant context
            context_data = retrieve_for_query(query)
            
            # Ensure KG results are valid
            kg_results = context_data.get('kg_results', [])
            if not isinstance(kg_results, list):
                logger.warning(f"KG results invalid: {kg_results}. Using empty list instead.")
                kg_results = []
            context_data['kg_results'] = kg_results
            
            # Build prompt
            prompt = self.build_prompt(query, context_data)
            
            # Generate LLM response
            llm_response = self.llm.generate(prompt)
            
            # Extract confidence
            confidence = self.extract_confidence(llm_response)
            
            # Prepare sources for response
            sources = []
            for source in context_data.get('sources', []):
                sources.append({
                    'id': source.get('id', 'unknown'),
                    'text': source.get('text', '')[:300] + "..." if len(source.get('text','')) > 300 else source.get('text',''),
                    'score': source.get('score', 0.0),
                    'type': source.get('source_type', 'unknown')
                })
            
            # Extract KG nodes mentioned safely
            kg_nodes = [node for node in kg_results if isinstance(node, str)]
            
            processing_time = time.time() - start_time
            
            response = AgentResponse(
                answer=llm_response,
                sources=sources,
                kg_nodes=kg_nodes,
                raw_llm_response=llm_response,
                query_type=context_data.get('query_type', 'unknown'),
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Return error response
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                sources=[],
                kg_nodes=[],
                raw_llm_response="",
                query_type="error",
                confidence=0.0,
                processing_time=time.time() - start_time
            )

class MockLLM:
    """Mock LLM for testing and offline usage."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response."""
        return """Based on the provided case studies, I can see several key patterns:

1. **Common Problems**: Customer service issues and employee turnover appear frequently
2. **Effective Solutions**: Training programs and improved benefits have shown positive results
3. **Key Insights**: Companies that invest in employee development see better customer satisfaction

**Sources**: Based on case study examples provided in the context.

This analysis is based on the available case study data, though more comprehensive data would provide deeper insights."""


# Global agent instance
_agent = None

def get_agent() -> CaseStudyAgent:
    """Get the global agent instance."""
    global _agent
    if _agent is None:
        try:
            _agent = CaseStudyAgent()
        except Exception as e:
            logger.warning(f"Failed to initialize LLM agent: {e}")
            logger.info("Using mock LLM for offline testing")
            # For testing/offline mode, create agent with mock LLM
            _agent = CaseStudyAgent.__new__(CaseStudyAgent)
            _agent.llm = MockLLM()
            _agent.retriever = get_retriever()
    return _agent


def answer_query(query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to answer queries.
    
    Args:
        query: User query
        user_context: Optional user context
        
    Returns:
        Response dictionary
    """
    agent = get_agent()
    response = agent.answer_query(query, user_context)
    
    return {
        'answer': response.answer,
        'sources': response.sources,
        'kg_nodes': response.kg_nodes,
        'raw_llm_response': response.raw_llm_response,
        'query_type': response.query_type,
        'confidence': response.confidence,
        'processing_time': response.processing_time
    }