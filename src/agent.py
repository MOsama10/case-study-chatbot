# """
# Agentic RAG implementation for case study analysis.
# Orchestrates retrieval and LLM reasoning.
# """

# from typing import Dict, Any, List, Optional
# import logging
# import time
# import json
# from dataclasses import dataclass

# # LLM API imports
# try:
#     import google.generativeai as genai
#     GENAI_AVAILABLE = True
# except ImportError:
#     GENAI_AVAILABLE = False

# try:
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# from .config import (
#     get_logger, GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
#     LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
# )
# from .retriever import get_retriever, retrieve_for_query
# from .professional_prompts import get_prompt_manager
# from .response_formatter import get_response_formatter
# from .direct_answer_handler import get_direct_answer_handler, handle_direct_question
# logger = get_logger(__name__)


# @dataclass
# class AgentResponse:
#     """Structured response from the agent."""
#     answer: str
#     sources: List[Dict[str, Any]]
#     kg_nodes: List[str]
#     raw_llm_response: str
#     query_type: str
#     confidence: float = 0.0
#     processing_time: float = 0.0


# class LLMWrapper:
#     """Wrapper for different LLM providers."""
    
#     def __init__(self, provider: str = LLM_PROVIDER):
#         """Initialize LLM wrapper with specified provider."""
#         self.provider = provider.lower()
#         self.client = None
#         self._initialize_client()
    
#     def _initialize_client(self) -> None:
#         """Initialize the appropriate LLM client."""
#         if self.provider == "gemini":
#             if not GENAI_AVAILABLE or not GEMINI_API_KEY:
#                 raise ValueError("Gemini API not available or API key not set")
            
#             import google.generativeai as genai
#             genai.configure(api_key=GEMINI_API_KEY)

#             # If model name not specified in config, pick latest supported generateContent model
#             try:
#                 from .config import GEMINI_MODEL
#             except ImportError:
#                 GEMINI_MODEL = None

#             if not GEMINI_MODEL:
#                 logger.info("No Gemini model set in config. Selecting latest supported model automatically...")
#                 available_models = genai.list_models()
#                 # Filter models that support generateContent
#                 valid_models = [m for m in available_models if 'generateContent' in m.supported_methods]
#                 if not valid_models:
#                     raise ValueError("No Gemini models supporting generateContent found.")
#                 # Pick the latest one by name (assuming naming reflects versions)
#                 GEMINI_MODEL = sorted(valid_models, key=lambda m: m.name, reverse=True)[0].name
#                 logger.info(f"Selected Gemini model: {GEMINI_MODEL}")

#             self.client = genai.GenerativeModel(GEMINI_MODEL)
#             logger.info(f"Initialized Gemini LLM client with model: {GEMINI_MODEL}")

#         elif self.provider == "openai":
#             if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
#                 raise ValueError("OpenAI API not available or API key not set")
#             self.client = OpenAI(api_key=OPENAI_API_KEY)
#             logger.info("Initialized OpenAI LLM client")

#         else:
#             raise ValueError(f"Unsupported LLM provider: {self.provider}")

    
#     def generate(self, prompt: str, **kwargs) -> str:
#         try:
#             if self.provider == "gemini":
#                 response = self.client.generate_content(
#                     prompt,
#                     generation_config=genai.types.GenerationConfig(
#                         temperature=kwargs.get('temperature', LLM_TEMPERATURE),
#                         max_output_tokens=kwargs.get('max_tokens', LLM_MAX_TOKENS),
#                     )
#                 )

#                 # Safely extract text
#                 if hasattr(response, 'text') and response.text:
#                     return response.text
#                 elif hasattr(response, 'candidates') and response.candidates:
#                     parts = getattr(response.candidates[0].content, 'parts', None)
#                     if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
#                         return parts[0].text
#                 # fallback if no valid content
#                 return "I apologize, but the LLM did not return a valid response."

#             elif self.provider == "openai":
#                 response = self.client.chat.completions.create(
#                     model=kwargs.get('model', 'gpt-3.5-turbo'),
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=kwargs.get('temperature', LLM_TEMPERATURE),
#                     max_tokens=kwargs.get('max_tokens', LLM_MAX_TOKENS),
#                     timeout=LLM_TIMEOUT
#                 )
#                 return response.choices[0].message.content

#         except Exception as e:
#             logger.error(f"LLM generation error: {e}")
#             return f"I apologize, but an error occurred during LLM generation: {str(e)}"



# class CaseStudyAgent:
#     """Main agent for case study analysis."""
    
#     def __init__(self, llm_provider: str = LLM_PROVIDER):
#         """Initialize the agent."""
#         self.llm = LLMWrapper(llm_provider)
#         self.retriever = get_retriever()
        
#         # Prompt templates
#         self.system_prompt = """You are an expert case study analyst. Your role is to analyze business case studies and provide insightful, evidence-based responses.

# Key responsibilities:
# 1. Analyze the provided context from case studies
# 2. Answer questions based on evidence from the sources
# 3. Identify patterns and relationships between problems and solutions
# 4. Provide actionable insights and recommendations
# 5. Always cite your sources using the provided source IDs

# Guidelines:
# - Be analytical and objective
# - Support your answers with specific evidence from the sources
# - Acknowledge when information is insufficient
# - Focus on actionable insights
# - Use clear, professional language"""

#         self.qa_template = """Context from case studies:
# {context}

# Question: {query}

# Please provide a comprehensive answer based on the case study context above. Include:
# 1. Direct answer to the question
# 2. Supporting evidence from the sources (cite source IDs)
# 3. Key insights or patterns identified
# 4. Any limitations or areas needing more information

# Answer:"""

#         self.analysis_template = """Context from case studies:
# {context}

# Analysis request: {query}

# Please provide a detailed analysis based on the case study context above. Include:
# 1. Overview of the situation/topic
# 2. Key problems or challenges identified
# 3. Solutions or approaches used
# 4. Results and outcomes
# 5. Patterns and insights
# 6. Recommendations for similar situations
# 7. Source citations for all claims

# Analysis:"""

#     def build_prompt(self, query: str, context_data: Dict[str, Any]) -> str:
#         """Build professional prompt based on query type."""
        
#         # Get professional prompt manager
#         prompt_manager = get_prompt_manager()
        
#         # Classify query type
#         query_type = self._classify_business_query(query)
        
#         # Enhance context with metadata
#         enhanced_context = prompt_manager.enhance_context_with_metadata(context_data)
#         enhanced_context['query'] = query
#         enhanced_context['query_type'] = query_type
        
#         # Generate professional prompt
#         return prompt_manager.get_professional_prompt(query_type, enhanced_context)

#     def _classify_business_query(self, query: str) -> str:
#         """Classify query for appropriate professional response."""
#         query_lower = query.lower()
        
#         if any(word in query_lower for word in ['analyze', 'analysis', 'examine']):
#             return 'analysis'
#         elif any(word in query_lower for word in ['recommend', 'suggest', 'strategy']):
#             return 'recommendation'
#         elif any(word in query_lower for word in ['compare', 'versus', 'vs']):
#             return 'comparison'
#         elif any(word in query_lower for word in ['trend', 'pattern', 'future']):
#             return 'trend'
#         elif any(word in query_lower for word in ['implement', 'execute', 'deploy']):
#             return 'implementation'
#         else:
#             return 'general'
    
#     def extract_confidence(self, response: str) -> float:
#         """
#         Extract confidence score from LLM response.
#         Simple heuristic-based approach.
        
#         Args:
#             response: LLM response text
            
#         Returns:
#             Confidence score between 0 and 1
#         """
#         # Simple heuristics for confidence
#         confidence_indicators = {
#             'certain': 0.9,
#             'confident': 0.8,
#             'likely': 0.7,
#             'probably': 0.6,
#             'possibly': 0.5,
#             'uncertain': 0.3,
#             'unclear': 0.2
#         }
        
#         response_lower = response.lower()
        
#         # Check for confidence indicators
#         max_confidence = 0.5  # Default
#         for indicator, score in confidence_indicators.items():
#             if indicator in response_lower:
#                 max_confidence = max(max_confidence, score)
        
#         # Adjust based on response length and detail
#         if len(response) > 500:
#             max_confidence += 0.1
        
#         # Check for source citations
#         if any(word in response_lower for word in ['source', 'according to', 'based on']):
#             max_confidence += 0.1
        
#         return min(max_confidence, 1.0)
#     def answer_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
#         """
#         Enhanced query answering with direct answer detection.
#         """
#         start_time = time.time()
        
#         try:
#             logger.info(f"Processing query: {query}")
            
#             # Retrieve relevant context first
#             context_data = retrieve_for_query(query)
            
#             # Check if this is a direct question that needs a simple answer
#             direct_handler = get_direct_answer_handler()
            
#             if direct_handler.is_direct_question(query):
#                 logger.info("Detected direct question - providing focused answer")
                
#                 # Get direct answer
#                 direct_response = handle_direct_question(query, context_data)
                
#                 if direct_response:
#                     # Calculate confidence for direct answer
#                     analysis = direct_handler.analyze_context_for_answer(query, context_data)
#                     confidence = analysis['confidence']
                    
#                     # Prepare sources for direct response
#                     sources = []
#                     for source in context_data.get('sources', []):
#                         sources.append({
#                             'id': source.get('id', 'unknown'),
#                             'text': source.get('text', '')[:200] + "..." if len(source.get('text','')) > 200 else source.get('text',''),
#                             'score': source.get('score', 0.0),
#                             'type': source.get('source_type', 'case_study')
#                         })
                    
#                     processing_time = time.time() - start_time
                    
#                     return AgentResponse(
#                         answer=direct_response,
#                         sources=sources,
#                         kg_nodes=context_data.get('kg_results', []),
#                         raw_llm_response=direct_response,
#                         query_type="direct_answer",
#                         confidence=confidence,
#                         processing_time=processing_time
#                     )
            
#             # If not a direct question, proceed with normal comprehensive analysis
#             # ... (rest of your existing answer_query method)
            
#             # Build prompt for comprehensive analysis
#             prompt = self.build_prompt(query, context_data)
            
#             # Generate LLM response
#             llm_response = self.llm.generate(prompt)
            
#             # Extract confidence
#             confidence = self.extract_confidence(llm_response)
            
#             # Prepare sources for response
#             sources = []
#             for source in context_data.get('sources', []):
#                 sources.append({
#                     'id': source.get('id', 'unknown'),
#                     'text': source.get('text', '')[:300] + "..." if len(source.get('text','')) > 300 else source.get('text',''),
#                     'score': source.get('score', 0.0),
#                     'type': source.get('source_type', 'unknown')
#                 })
            
#             kg_nodes = [node for node in context_data.get('kg_results', []) if isinstance(node, str)]
            
#             processing_time = time.time() - start_time
            
#             response = AgentResponse(
#                 answer=llm_response,
#                 sources=sources,
#                 kg_nodes=kg_nodes,
#                 raw_llm_response=llm_response,
#                 query_type=context_data.get('query_type', 'comprehensive'),
#                 confidence=confidence,
#                 processing_time=processing_time
#             )
            
#             logger.info(f"Query processed successfully in {processing_time:.2f}s")
#             return response
            
#         except Exception as e:
#             logger.error(f"Error processing query: {e}")
            
#             # Return error response
#             return AgentResponse(
#                 answer=f"I apologize, but I encountered an error while processing your query: {str(e)}",
#                 sources=[],
#                 kg_nodes=[],
#                 raw_llm_response="",
#                 query_type="error",
#                 confidence=0.0,
#                 processing_time=time.time() - start_time
#             )

# class MockLLM:
#     """Mock LLM for testing and offline usage."""
    
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate mock response."""
#         return """Based on the provided case studies, I can see several key patterns:

# 1. **Common Problems**: Customer service issues and employee turnover appear frequently
# 2. **Effective Solutions**: Training programs and improved benefits have shown positive results
# 3. **Key Insights**: Companies that invest in employee development see better customer satisfaction

# **Sources**: Based on case study examples provided in the context.

# This analysis is based on the available case study data, though more comprehensive data would provide deeper insights."""


# # Global agent instance
# _agent = None

# def get_agent() -> CaseStudyAgent:
#     """Get the global agent instance."""
#     global _agent
#     if _agent is None:
#         try:
#             _agent = CaseStudyAgent()
#         except Exception as e:
#             logger.warning(f"Failed to initialize LLM agent: {e}")
#             logger.info("Using mock LLM for offline testing")
#             # For testing/offline mode, create agent with mock LLM
#             _agent = CaseStudyAgent.__new__(CaseStudyAgent)
#             _agent.llm = MockLLM()
#             _agent.retriever = get_retriever()
#     return _agent


# def answer_query(query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """
#     Convenience function to answer queries.
    
#     Args:
#         query: User query
#         user_context: Optional user context
        
#     Returns:
#         Response dictionary
#     """
#     agent = get_agent()
#     response = agent.answer_query(query, user_context)
    
#     return {
#         'answer': response.answer,
#         'sources': response.sources,
#         'kg_nodes': response.kg_nodes,
#         'raw_llm_response': response.raw_llm_response,
#         'query_type': response.query_type,
#         'confidence': response.confidence,
#         'processing_time': response.processing_time
#     }



# """
# Agentic RAG implementation for legal document analysis.
# Simplified version focused on arbitration and legal documents.
# """

# from typing import Dict, Any, List, Optional
# import logging
# import time
# from dataclasses import dataclass

# # LLM API imports
# try:
#     import google.generativeai as genai
#     GENAI_AVAILABLE = True
# except ImportError:
#     GENAI_AVAILABLE = False

# try:
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# from .config import (
#     get_logger, GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
#     LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
# )
# from .retriever import get_retriever, retrieve_for_query
# from .direct_answer_handler import get_direct_answer_handler, handle_direct_question
# from .response_formatter import get_response_formatter

# logger = get_logger(__name__)


# @dataclass
# class AgentResponse:
#     """Structured response from the agent."""
#     answer: str
#     sources: List[Dict[str, Any]]
#     kg_nodes: List[str]
#     raw_llm_response: str
#     query_type: str
#     confidence: float = 0.0
#     processing_time: float = 0.0


# class LLMWrapper:
#     """Wrapper for different LLM providers."""
    
#     def __init__(self, provider: str = LLM_PROVIDER):
#         """Initialize LLM wrapper with specified provider."""
#         self.provider = provider.lower()
#         self.client = None
#         self._initialize_client()
    
#     def _initialize_client(self) -> None:
#         """Initialize the appropriate LLM client."""
#         if self.provider == "gemini":
#             if not GENAI_AVAILABLE or not GEMINI_API_KEY:
#                 raise ValueError("Gemini API not available or API key not set")
            
#             import google.generativeai as genai
#             genai.configure(api_key=GEMINI_API_KEY)

#             # Use a default model if not specified
#             try:
#                 from .config import GEMINI_MODEL
#             except ImportError:
#                 GEMINI_MODEL = "gemini-pro"

#             self.client = genai.GenerativeModel(GEMINI_MODEL)
#             logger.info(f"Initialized Gemini LLM client with model: {GEMINI_MODEL}")

#         elif self.provider == "openai":
#             if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
#                 raise ValueError("OpenAI API not available or API key not set")
#             self.client = OpenAI(api_key=OPENAI_API_KEY)
#             logger.info("Initialized OpenAI LLM client")

#         else:
#             raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate text using the LLM."""
#         try:
#             if self.provider == "gemini":
#                 response = self.client.generate_content(
#                     prompt,
#                     generation_config=genai.types.GenerationConfig(
#                         temperature=kwargs.get('temperature', LLM_TEMPERATURE),
#                         max_output_tokens=kwargs.get('max_tokens', LLM_MAX_TOKENS),
#                     )
#                 )

#                 # Safely extract text
#                 if hasattr(response, 'text') and response.text:
#                     return response.text
#                 elif hasattr(response, 'candidates') and response.candidates:
#                     parts = getattr(response.candidates[0].content, 'parts', None)
#                     if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
#                         return parts[0].text
                
#                 return "I apologize, but I couldn't generate a response at this time."

#             elif self.provider == "openai":
#                 response = self.client.chat.completions.create(
#                     model=kwargs.get('model', 'gpt-3.5-turbo'),
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=kwargs.get('temperature', LLM_TEMPERATURE),
#                     max_tokens=kwargs.get('max_tokens', LLM_MAX_TOKENS),
#                     timeout=LLM_TIMEOUT
#                 )
#                 return response.choices[0].message.content

#         except Exception as e:
#             logger.error(f"LLM generation error: {e}")
#             return f"I apologize, but an error occurred while generating the response: {str(e)}"


# class LegalDocumentAgent:
#     """Main agent for legal document and arbitration analysis."""
    
#     def __init__(self, llm_provider: str = LLM_PROVIDER):
#         """Initialize the agent."""
#         self.llm = LLMWrapper(llm_provider)
#         self.retriever = get_retriever()
        
#         # Simple system prompt for legal documents
#         self.system_prompt = """You are a legal document analysis expert specializing in arbitration cases, legal procedures, and decision trees. Your role is to analyze legal documents and provide clear, accurate responses based on the evidence.

# Key responsibilities:
# 1. Analyze legal documents and arbitration cases
# 2. Provide accurate information about legal procedures and timelines
# 3. Explain arbitration processes and decision trees
# 4. Reference specific documents and case precedents
# 5. Always cite your sources using the provided document IDs

# Guidelines:
# - Be precise and factual in legal matters
# - Support answers with specific evidence from legal documents
# - Acknowledge limitations when information is insufficient
# - Use clear, professional legal language
# - Focus on practical, actionable legal guidance"""

#     def build_simple_prompt(self, query: str, context_data: Dict[str, Any]) -> str:
#         """Build a simple prompt for legal document analysis."""
        
#         # Get context information
#         context_text = context_data.get('context_text', '')
#         query_type = context_data.get('query_type', 'general')
#         sources = context_data.get('sources', [])
        
#         # Build prompt based on query type
#         if query_type == 'analysis':
#             prompt_template = """Context from Legal Documents:
# {context}

# Legal Analysis Request: {query}

# Please provide a comprehensive legal analysis based on the document context above. Include:
# 1. Direct answer to the legal question
# 2. Supporting evidence from the legal documents (cite document IDs)
# 3. Relevant legal procedures or precedents
# 4. Any limitations or areas needing additional legal research

# Legal Analysis:"""

#         elif query_type == 'recommendation':
#             prompt_template = """Context from Legal Documents:
# {context}

# Legal Recommendation Request: {query}

# Please provide legal recommendations based on the document context above. Include:
# 1. Recommended legal approach or action
# 2. Supporting evidence from legal documents (cite document IDs)
# 3. Legal precedents or procedures that support this recommendation
# 4. Any risks or considerations to be aware of

# Legal Recommendation:"""

#         elif query_type == 'direct_answer':
#             prompt_template = """Context from Legal Documents:
# {context}

# Direct Legal Question: {query}

# Please provide a direct, clear answer based on the legal documents above. Include:
# 1. Clear, direct answer to the question
# 2. Supporting evidence from the documents (cite document IDs)
# 3. Any relevant legal procedures or timelines

# Direct Answer:"""

#         else:
#             prompt_template = """Context from Legal Documents:
# {context}

# Legal Question: {query}

# Please provide a clear response based on the legal document context above. Include:
# 1. Answer to the legal question
# 2. Supporting evidence from the documents (cite document IDs)
# 3. Relevant legal information or procedures
# 4. Any important considerations or limitations

# Response:"""

#         return self.system_prompt + "\n\n" + prompt_template.format(
#             context=context_text,
#             query=query
#         )
    
#     def extract_confidence(self, response: str, sources: List[Dict]) -> float:
#         """Extract confidence score based on response quality and sources."""
#         base_confidence = 0.5
        
#         # Boost confidence based on response length and detail
#         if len(response) > 200:
#             base_confidence += 0.1
#         if len(response) > 400:
#             base_confidence += 0.1
        
#         # Boost confidence based on source quality
#         if sources:
#             avg_source_score = sum(s.get('score', 0) for s in sources) / len(sources)
#             base_confidence += avg_source_score * 0.3
        
#         # Check for legal terminology (indicates relevant content)
#         legal_terms = ['arbitration', 'legal', 'court', 'procedure', 'clause', 'decision', 'case']
#         legal_matches = sum(1 for term in legal_terms if term.lower() in response.lower())
#         if legal_matches > 0:
#             base_confidence += min(legal_matches * 0.05, 0.2)
        
#         # Check for source citations
#         if 'document' in response.lower() or 'source' in response.lower():
#             base_confidence += 0.1
        
#         return min(base_confidence, 1.0)
    
#     def answer_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
#         """Answer legal document queries."""
#         start_time = time.time()
        
#         try:
#             logger.info(f"Processing legal query: {query}")
            
#             # Retrieve relevant context
#             context_data = retrieve_for_query(query)
            
#             # Check if this is a direct question
#             direct_handler = get_direct_answer_handler()
            
#             if direct_handler.is_direct_question(query):
#                 logger.info("Detected direct legal question")
                
#                 # Try to get direct answer
#                 direct_response = handle_direct_question(query, context_data)
                
#                 if direct_response:
#                     # Calculate confidence for direct answer
#                     analysis = direct_handler.analyze_context_for_answer(query, context_data)
#                     confidence = analysis['confidence']
                    
#                     # Prepare sources
#                     sources = []
#                     for source in context_data.get('sources', []):
#                         sources.append({
#                             'id': source.get('id', 'unknown'),
#                             'text': source.get('text', '')[:200] + "..." if len(source.get('text','')) > 200 else source.get('text',''),
#                             'score': source.get('score', 0.0),
#                             'type': source.get('source_type', 'legal_document')
#                         })
                    
#                     processing_time = time.time() - start_time
                    
#                     return AgentResponse(
#                         answer=direct_response,
#                         sources=sources,
#                         kg_nodes=context_data.get('kg_results', []),
#                         raw_llm_response=direct_response,
#                         query_type="direct_answer",
#                         confidence=confidence,
#                         processing_time=processing_time
#                     )
            
#             # Build prompt for comprehensive analysis
#             prompt = self.build_simple_prompt(query, context_data)
            
#             # Generate LLM response
#             llm_response = self.llm.generate(prompt)
            
#             # Extract confidence
#             confidence = self.extract_confidence(llm_response, context_data.get('sources', []))
            
#             # Prepare sources for response
#             sources = []
#             for source in context_data.get('sources', []):
#                 sources.append({
#                     'id': source.get('id', 'unknown'),
#                     'text': source.get('text', '')[:300] + "..." if len(source.get('text','')) > 300 else source.get('text',''),
#                     'score': source.get('score', 0.0),
#                     'type': source.get('source_type', 'legal_document')
#                 })
            
#             kg_nodes = [node for node in context_data.get('kg_results', []) if isinstance(node, str)]
            
#             processing_time = time.time() - start_time
            
#             response = AgentResponse(
#                 answer=llm_response,
#                 sources=sources,
#                 kg_nodes=kg_nodes,
#                 raw_llm_response=llm_response,
#                 query_type=context_data.get('query_type', 'legal_analysis'),
#                 confidence=confidence,
#                 processing_time=processing_time
#             )
            
#             logger.info(f"Legal query processed successfully in {processing_time:.2f}s")
#             return response
            
#         except Exception as e:
#             logger.error(f"Error processing legal query: {e}")
            
#             # Return error response
#             return AgentResponse(
#                 answer=f"I apologize, but I encountered an error while analyzing the legal documents: {str(e)}",
#                 sources=[],
#                 kg_nodes=[],
#                 raw_llm_response="",
#                 query_type="error",
#                 confidence=0.0,
#                 processing_time=time.time() - start_time
#             )


# class MockLLM:
#     """Mock LLM for testing when API is not available."""
    
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate mock legal response."""
#         return """Based on the legal documents and arbitration cases available:

# **Legal Analysis:**
# The documents indicate that arbitration procedures typically follow established timelines and cost structures. Key considerations include:

# 1. **Procedural Requirements**: Standard arbitration processes require proper notice, valid arbitration clauses, and adherence to established timelines.

# 2. **Cost Considerations**: Arbitration costs vary based on claim value and complexity, with typical ranges from $50,000 to $500,000.

# 3. **Legal Precedents**: The cases show that successful arbitration requires clear documentation and proper procedural compliance.

# **Sources**: Based on legal documents and arbitration case precedents in the document collection.

# This analysis is based on the available legal documents, though additional research may be needed for specific legal advice."""


# # Global agent instance
# _agent = None

# def get_agent() -> LegalDocumentAgent:
#     """Get the global agent instance."""
#     global _agent
#     if _agent is None:
#         try:
#             _agent = LegalDocumentAgent()
#         except Exception as e:
#             logger.warning(f"Failed to initialize LLM agent: {e}")
#             logger.info("Using mock LLM for testing")
#             # For testing/offline mode, create agent with mock LLM
#             _agent = LegalDocumentAgent.__new__(LegalDocumentAgent)
#             _agent.llm = MockLLM()
#             _agent.retriever = get_retriever()
#     return _agent


# def answer_query(query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """
#     Convenience function to answer legal queries.
    
#     Args:
#         query: User query about legal documents
#         user_context: Optional user context
        
#     Returns:
#         Response dictionary
#     """
#     agent = get_agent()
#     response = agent.answer_query(query, user_context)
    
#     return {
#         'answer': response.answer,
#         'sources': response.sources,
#         'kg_nodes': response.kg_nodes,
#         'raw_llm_response': response.raw_llm_response,
#         'query_type': response.query_type,
#         'confidence': response.confidence,
#         'processing_time': response.processing_time
#     }

###########################3
# """
# Agentic RAG implementation for legal document analysis.
# Simplified version focused on arbitration and legal documents.
# """

# from typing import Dict, Any, List, Optional
# import logging
# import time
# from dataclasses import dataclass

# # LLM API imports
# try:
#     import google.generativeai as genai
#     GENAI_AVAILABLE = True
# except ImportError:
#     GENAI_AVAILABLE = False

# try:
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# from .config import (
#     get_logger, GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
#     LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT, 
#     LEGAL_LLM_TEMPERATURE, LEGAL_LLM_MAX_TOKENS,
#     LEGAL_RESPONSE_MIN_LENGTH, LEGAL_RESPONSE_MAX_LENGTH
# )
# from .retriever import get_retriever, retrieve_for_query
# from .direct_answer_handler import get_direct_answer_handler, handle_direct_question
# from .response_formatter import get_response_formatter

# logger = get_logger(__name__)


# @dataclass
# class AgentResponse:
#     """Structured response from the agent."""
#     answer: str
#     sources: List[Dict[str, Any]]
#     kg_nodes: List[str]
#     raw_llm_response: str
#     query_type: str
#     confidence: float = 0.0
#     processing_time: float = 0.0


# class LLMWrapper:
#     """Wrapper for different LLM providers."""
    
#     def __init__(self, provider: str = LLM_PROVIDER):
#         """Initialize LLM wrapper with specified provider."""
#         self.provider = provider.lower()
#         self.client = None
#         self._initialize_client()
    
#     def _initialize_client(self) -> None:
#         """Initialize the appropriate LLM client."""
#         if self.provider == "gemini":
#             if not GENAI_AVAILABLE or not GEMINI_API_KEY:
#                 raise ValueError("Gemini API not available or API key not set")
            
#             import google.generativeai as genai
#             genai.configure(api_key=GEMINI_API_KEY)

#             # Use a default model if not specified
#             try:
#                 from .config import GEMINI_MODEL
#             except ImportError:
#                 GEMINI_MODEL = "gemini-pro"

#             self.client = genai.GenerativeModel(GEMINI_MODEL)
#             logger.info(f"Initialized Gemini LLM client with model: {GEMINI_MODEL}")

#         elif self.provider == "openai":
#             if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
#                 raise ValueError("OpenAI API not available or API key not set")
#             self.client = OpenAI(api_key=OPENAI_API_KEY)
#             logger.info("Initialized OpenAI LLM client")

#         else:
#             raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate text using the LLM with higher token limits for legal content."""
#         try:
#             # Use higher token limits for legal content
#             max_tokens = kwargs.get('max_tokens', LEGAL_LLM_MAX_TOKENS)
#             temperature = kwargs.get('temperature', LEGAL_LLM_TEMPERATURE)
            
#             if self.provider == "gemini":
#                 response = self.client.generate_content(
#                     prompt,
#                     generation_config=genai.types.GenerationConfig(
#                         temperature=temperature,
#                         max_output_tokens=max_tokens,
#                     )
#                 )

#                 # Safely extract text
#                 if hasattr(response, 'text') and response.text:
#                     return response.text
#                 elif hasattr(response, 'candidates') and response.candidates:
#                     parts = getattr(response.candidates[0].content, 'parts', None)
#                     if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
#                         return parts[0].text
                
#                 return "I apologize, but I couldn't generate a complete response at this time."

#             elif self.provider == "openai":
#                 response = self.client.chat.completions.create(
#                     model=kwargs.get('model', 'gpt-4o-mini'),
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     timeout=LLM_TIMEOUT
#                 )
#                 return response.choices[0].message.content

#         except Exception as e:
#             logger.error(f"LLM generation error: {e}")
#             return f"I apologize, but an error occurred while generating the response. Please try rephrasing your question or try again."


# class LegalDocumentAgent:
#     """Main agent for legal document and arbitration analysis."""
    
#     def __init__(self, llm_provider: str = LLM_PROVIDER):
#         """Initialize the agent."""
#         self.llm = LLMWrapper(llm_provider)
#         self.retriever = get_retriever()
        
#         # Simple system prompt for legal documents
#         self.system_prompt = """You are a legal document analysis expert specializing in arbitration cases, legal procedures, and decision trees. Your role is to analyze legal documents and provide comprehensive, detailed responses based on the evidence.

# Key responsibilities:
# 1. Analyze legal documents and arbitration cases thoroughly
# 2. Provide complete, detailed information about legal procedures and timelines
# 3. Explain arbitration processes and decision trees comprehensively
# 4. Reference specific documents and case precedents with full context
# 5. Always cite your sources using the provided document IDs
# 6. Provide complete answers - do not truncate or summarize excessively

# Guidelines:
# - Be precise, detailed, and comprehensive in legal matters
# - Provide full explanations with complete reasoning
# - Support answers with extensive evidence from legal documents
# - Include all relevant legal considerations and implications
# - Use clear, professional legal language with full explanations
# - Ensure responses are complete and address all aspects of the question
# - When listing items or steps, include all relevant points
# - Do not cut responses short - provide thorough analysis"""

#     def build_simple_prompt(self, query: str, context_data: Dict[str, Any]) -> str:
#         """Build a simple prompt for legal document analysis."""
        
#         # Get context information
#         context_text = context_data.get('context_text', '')
#         query_type = context_data.get('query_type', 'general')
#         sources = context_data.get('sources', [])
        
#         # Build prompt based on query type
#         if query_type == 'analysis':
#             prompt_template = """Context from Legal Documents:
# {context}

# Legal Analysis Request: {query}

# Please provide a comprehensive legal analysis based on the document context above. Include:
# 1. Direct answer to the legal question
# 2. Supporting evidence from the legal documents (cite document IDs)
# 3. Relevant legal procedures or precedents
# 4. Any limitations or areas needing additional legal research

# Legal Analysis:"""

#         elif query_type == 'recommendation':
#             prompt_template = """Context from Legal Documents:
# {context}

# Legal Recommendation Request: {query}

# Please provide legal recommendations based on the document context above. Include:
# 1. Recommended legal approach or action
# 2. Supporting evidence from legal documents (cite document IDs)
# 3. Legal precedents or procedures that support this recommendation
# 4. Any risks or considerations to be aware of

# Legal Recommendation:"""

#         elif query_type == 'direct_answer':
#             prompt_template = """Context from Legal Documents:
# {context}

# Direct Legal Question: {query}

# Please provide a direct, clear answer based on the legal documents above. Include:
# 1. Clear, direct answer to the question
# 2. Supporting evidence from the documents (cite document IDs)
# 3. Any relevant legal procedures or timelines

# Direct Answer:"""

#         else:
#             prompt_template = """Context from Legal Documents:
# {context}

# Legal Question: {query}

# Please provide a clear response based on the legal document context above. Include:
# 1. Answer to the legal question
# 2. Supporting evidence from the documents (cite document IDs)
# 3. Relevant legal information or procedures
# 4. Any important considerations or limitations

# Response:"""

#         return self.system_prompt + "\n\n" + prompt_template.format(
#             context=context_text,
#             query=query
#         )
    
#     def extract_confidence(self, response: str, sources: List[Dict]) -> float:
#         """Extract confidence score based on response quality and sources."""
#         base_confidence = 0.5
        
#         # Boost confidence based on response length and detail
#         if len(response) > 200:
#             base_confidence += 0.1
#         if len(response) > 400:
#             base_confidence += 0.1
        
#         # Boost confidence based on source quality
#         if sources:
#             avg_source_score = sum(s.get('score', 0) for s in sources) / len(sources)
#             base_confidence += avg_source_score * 0.3
        
#         # Check for legal terminology (indicates relevant content)
#         legal_terms = ['arbitration', 'legal', 'court', 'procedure', 'clause', 'decision', 'case']
#         legal_matches = sum(1 for term in legal_terms if term.lower() in response.lower())
#         if legal_matches > 0:
#             base_confidence += min(legal_matches * 0.05, 0.2)
        
#         # Check for source citations
#         if 'document' in response.lower() or 'source' in response.lower():
#             base_confidence += 0.1
        
#         return min(base_confidence, 1.0)
    
#     def answer_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
#         """Answer legal document queries."""
#         start_time = time.time()
        
#         try:
#             logger.info(f"Processing legal query: {query}")
            
#             # Retrieve relevant context
#             context_data = retrieve_for_query(query)
            
#             # Check if this is a direct question
#             direct_handler = get_direct_answer_handler()
            
#             if direct_handler.is_direct_question(query):
#                 logger.info("Detected direct legal question")
                
#                 # Try to get direct answer
#                 direct_response = handle_direct_question(query, context_data)
                
#                 if direct_response:
#                     # Calculate confidence for direct answer
#                     analysis = direct_handler.analyze_context_for_answer(query, context_data)
#                     confidence = analysis['confidence']
                    
#                     # Prepare sources
#                     sources = []
#                     for source in context_data.get('sources', []):
#                         sources.append({
#                             'id': source.get('id', 'unknown'),
#                             'text': source.get('text', '')[:200] + "..." if len(source.get('text','')) > 200 else source.get('text',''),
#                             'score': source.get('score', 0.0),
#                             'type': source.get('source_type', 'legal_document')
#                         })
                    
#                     processing_time = time.time() - start_time
                    
#                     return AgentResponse(
#                         answer=direct_response,
#                         sources=sources,
#                         kg_nodes=context_data.get('kg_results', []),
#                         raw_llm_response=direct_response,
#                         query_type="direct_answer",
#                         confidence=confidence,
#                         processing_time=processing_time
#                     )
            
#             # Build prompt for comprehensive analysis
#             prompt = self.build_simple_prompt(query, context_data)
            
#             # Generate LLM response
#             llm_response = self.llm.generate(prompt)
            
#             # Extract confidence
#             confidence = self.extract_confidence(llm_response, context_data.get('sources', []))
            
#             # Prepare sources for response
#             sources = []
#             for source in context_data.get('sources', []):
#                 sources.append({
#                     'id': source.get('id', 'unknown'),
#                     'text': source.get('text', '')[:300] + "..." if len(source.get('text','')) > 300 else source.get('text',''),
#                     'score': source.get('score', 0.0),
#                     'type': source.get('source_type', 'legal_document')
#                 })
            
#             kg_nodes = [node for node in context_data.get('kg_results', []) if isinstance(node, str)]
            
#             processing_time = time.time() - start_time
            
#             response = AgentResponse(
#                 answer=llm_response,
#                 sources=sources,
#                 kg_nodes=kg_nodes,
#                 raw_llm_response=llm_response,
#                 query_type=context_data.get('query_type', 'legal_analysis'),
#                 confidence=confidence,
#                 processing_time=processing_time
#             )
            
#             logger.info(f"Legal query processed successfully in {processing_time:.2f}s")
#             return response
            
#         except Exception as e:
#             logger.error(f"Error processing legal query: {e}")
            
#             # Return error response
#             return AgentResponse(
#                 answer=f"I apologize, but I encountered an error while analyzing the legal documents: {str(e)}",
#                 sources=[],
#                 kg_nodes=[],
#                 raw_llm_response="",
#                 query_type="error",
#                 confidence=0.0,
#                 processing_time=time.time() - start_time
#             )


# class MockLLM:
#     """Mock LLM for testing when API is not available."""
    
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate mock legal response."""
#         return """Based on the legal documents and arbitration cases available:

# **Legal Analysis:**
# The documents indicate that arbitration procedures typically follow established timelines and cost structures. Key considerations include:

# 1. **Procedural Requirements**: Standard arbitration processes require proper notice, valid arbitration clauses, and adherence to established timelines.

# 2. **Cost Considerations**: Arbitration costs vary based on claim value and complexity, with typical ranges from $50,000 to $500,000.

# 3. **Legal Precedents**: The cases show that successful arbitration requires clear documentation and proper procedural compliance.

# **Sources**: Based on legal documents and arbitration case precedents in the document collection.

# This analysis is based on the available legal documents, though additional research may be needed for specific legal advice."""


# # Global agent instance
# _agent = None

# def get_agent() -> LegalDocumentAgent:
#     """Get the global agent instance."""
#     global _agent
#     if _agent is None:
#         try:
#             _agent = LegalDocumentAgent()
#         except Exception as e:
#             logger.warning(f"Failed to initialize LLM agent: {e}")
#             logger.info("Using mock LLM for testing")
#             # For testing/offline mode, create agent with mock LLM
#             _agent = LegalDocumentAgent.__new__(LegalDocumentAgent)
#             _agent.llm = MockLLM()
#             _agent.retriever = get_retriever()
#     return _agent


# def answer_query(query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """
#     Convenience function to answer legal queries.
    
#     Args:
#         query: User query about legal documents
#         user_context: Optional user context
        
#     Returns:
#         Response dictionary
#     """
#     agent = get_agent()
#     response = agent.answer_query(query, user_context)
    
#     return {
#         'answer': response.answer,
#         'sources': response.sources,
#         'kg_nodes': response.kg_nodes,
#         'raw_llm_response': response.raw_llm_response,
#         'query_type': response.query_type,
#         'confidence': response.confidence,
#         'processing_time': response.processing_time
#     }


"""
Legal Professional Agent - Humanized responses with summary-style answers.
Acts as an experienced legal counsel specializing in arbitration and commercial law.
"""

from typing import Dict, Any, List, Optional
import logging
import time
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
from .direct_answer_handler import get_direct_answer_handler, handle_direct_question
from .response_formatter import get_response_formatter

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
            if not GENAI_AVAILABLE or not GEMINI_API_KEY:
                raise ValueError("Gemini API not available or API key not set")
            
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel("gemini-pro")
            logger.info("Legal professional assistant initialized with Gemini")

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
                raise ValueError("OpenAI API not available or API key not set")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("Legal professional assistant initialized with OpenAI")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_legal_response(self, prompt: str, **kwargs) -> str:
        """Generate a humanized legal professional response."""
        try:
            # Use optimal settings for legal responses
            max_tokens = kwargs.get('max_tokens', 3000)  # Shorter for summary style
            temperature = kwargs.get('temperature', 0.3)  # Balanced for personality
            
            if self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )

                if hasattr(response, 'text') and response.text:
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    parts = getattr(response.candidates[0].content, 'parts', None)
                    if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
                        return parts[0].text
                
                return "I apologize, but I'm having trouble accessing the legal documents right now. Could you please rephrase your question?"

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=kwargs.get('model', 'gpt-4o-mini'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Legal LLM generation error: {e}")
            return "I'm experiencing some technical difficulties accessing the legal database. Let me try a different approach to your question."


class LegalProfessionalAgent:
    """Main agent that acts as an experienced legal professional."""
    
    def __init__(self, llm_provider: str = LLM_PROVIDER):
        """Initialize the legal professional agent."""
        self.llm = LegalProfessionalLLM(llm_provider)
        self.retriever = get_retriever()
        
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
        answer_focus = context_data.get('answer_focus', '')
        
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
            context_data = retrieve_for_query(query)
            
            # Check for direct legal questions first
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
            _legal_agent.retriever = get_retriever()
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