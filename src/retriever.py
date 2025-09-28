
"""
Answer-focused retrieval system for legal documents.
Prioritizes finding actual answers and specific information from documents.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import re
import numpy as np
from collections import defaultdict

from src.config import get_logger, VECTOR_TOP_K, VECTOR_SIMILARITY_THRESHOLD
from src.embeddings import get_embedding_manager, search_similar

# Knowledge Graph import with fallback
try:
    from src.knowledge_graph import get_knowledge_graph, query_kg
    KG_AVAILABLE = True
except ImportError:
    try:
        from src.knowledge_graph import load_kg, query_kg
        def get_knowledge_graph():
            return load_kg()
        KG_AVAILABLE = True
    except ImportError:
        import networkx as nx
        def get_knowledge_graph():
            return nx.Graph()
        def query_kg(query, kg=None, include_neighbors=True):
            return []
        KG_AVAILABLE = False

logger = get_logger(__name__)


class AnswerFocusedQueryProcessor:
    """Processes queries to focus on finding specific answers."""
    
    def __init__(self):
        """Initialize with answer-focused patterns."""
        
        # Answer-seeking question patterns
        self.answer_patterns = {
            'definition': [
                r'what is\s+(.+)',
                r'what are\s+(.+)', 
                r'define\s+(.+)',
                r'explain\s+(.+)',
                r'describe\s+(.+)'
            ],
            'procedure': [
                r'how\s+(?:do|does|to)\s+(.+)',
                r'what\s+(?:is\s+the\s+)?process\s+(?:for|of)\s+(.+)',
                r'what\s+(?:are\s+the\s+)?steps\s+(?:for|to)\s+(.+)',
                r'how\s+long\s+(?:does|is)\s+(.+)',
                r'what\s+(?:is\s+the\s+)?timeline\s+(?:for|of)\s+(.+)'
            ],
            'factual': [
                r'(?:is\s+there|are\s+there)\s+(.+)',
                r'(?:does\s+the|can\s+the)\s+(.+)',
                r'when\s+(?:is|does|did)\s+(.+)',
                r'where\s+(?:is|does|did)\s+(.+)',
                r'who\s+(?:is|does|did)\s+(.+)'
            ],
            'cost_time': [
                r'how\s+much\s+(?:does|is|cost)\s+(.+)',
                r'what\s+(?:is\s+the\s+)?cost\s+(?:of|for)\s+(.+)',
                r'how\s+long\s+(?:does|takes?)\s+(.+)',
                r'what\s+(?:is\s+the\s+)?fee\s+(?:for|of)\s+(.+)'
            ],
            'legal_specific': [
                r'(?:is\s+the\s+)?arbitration\s+clause\s+(.+)',
                r'what\s+(?:is\s+the\s+)?validity\s+(?:of|for)\s+(.+)',
                r'(?:can|may)\s+(?:the\s+)?tribunal\s+(.+)',
                r'what\s+(?:are\s+the\s+)?requirements\s+(?:for|of)\s+(.+)'
            ]
        }
        
        # Answer indicator keywords in documents
        self.answer_indicators = [
            # Direct answer indicators
            'answer:', 'response:', 'solution:', 'result:',
            'yes,', 'no,', 'according to', 'based on',
            'the answer is', 'the result is', 'it is determined',
            
            # Legal answer indicators  
            'the clause states', 'the agreement provides', 'the procedure requires',
            'the tribunal may', 'the arbitration rules state', 'the law provides',
            'validity requires', 'enforcement depends on', 'the requirement is',
            
            # Factual indicators
            'the cost is', 'the timeline is', 'the deadline is',
            'the fee amounts to', 'the duration is', 'the process takes',
            
            # Conclusion indicators
            'therefore,', 'thus,', 'consequently,', 'as a result,',
            'in conclusion,', 'finally,', 'ultimately,'
        ]
        
        # Keywords that indicate specific information
        self.specific_info_keywords = [
            'specific', 'exactly', 'precisely', 'clearly', 'explicitly',
            'stated', 'defined', 'specified', 'outlined', 'detailed',
            'amount', 'number', 'percentage', 'days', 'months', 'years',
            'deadline', 'timeline', 'procedure', 'requirement', 'condition'
        ]
    
    def classify_answer_intent(self, query: str) -> str:
        """Classify what type of answer the query is seeking."""
        query_lower = query.lower()
        
        for intent_type, patterns in self.answer_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent_type
        
        return 'general_answer'
    
    def extract_answer_focus(self, query: str) -> str:
        """Extract the main focus of what answer is being sought."""
        query_lower = query.lower()
        
        # Try to extract the key topic from question patterns
        for patterns in self.answer_patterns.values():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    focus = match.group(1).strip()
                    # Clean up common question words
                    focus = re.sub(r'\b(a|an|the|for|of|in|on|at|to|by|with)\b', ' ', focus)
                    return focus.strip()
        
        # Fallback: extract key legal/arbitration terms
        legal_terms = [
            'arbitration', 'clause', 'agreement', 'contract', 'tribunal',
            'procedure', 'timeline', 'cost', 'fee', 'deadline', 'requirement',
            'validity', 'enforcement', 'jurisdiction', 'award', 'ruling'
        ]
        
        found_terms = [term for term in legal_terms if term in query_lower]
        if found_terms:
            return ' '.join(found_terms)
        
        # Ultimate fallback: return cleaned query
        return re.sub(r'\b(what|how|when|where|who|why|is|are|does|can|may)\b', '', query_lower).strip()


class AnswerFocusedRetriever:
    """Enhanced retriever that focuses on finding specific answers."""
    
    def __init__(self):
        """Initialize the answer-focused retriever."""
        self.embedding_manager = get_embedding_manager()
        
        try:
            self.knowledge_graph = get_knowledge_graph()
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")
            import networkx as nx
            self.knowledge_graph = nx.Graph()
        
        self.query_processor = AnswerFocusedQueryProcessor()
    
    def answer_focused_vector_search(self, query: str, answer_intent: str, top_k: int = VECTOR_TOP_K) -> List[Tuple[str, float, str]]:
        """Vector search optimized for finding answers."""
        
        # Extract the main focus of the answer being sought
        answer_focus = self.query_processor.extract_answer_focus(query)
        
        # Create answer-focused search queries
        search_queries = [
            query,  # Original query
            answer_focus,  # Extracted focus
        ]
        
        # Add intent-specific search variations
        if answer_intent == 'definition':
            search_queries.extend([
                f"{answer_focus} definition",
                f"what is {answer_focus}",
                f"{answer_focus} means"
            ])
        elif answer_intent == 'procedure':
            search_queries.extend([
                f"{answer_focus} procedure",
                f"{answer_focus} process",
                f"how to {answer_focus}",
                f"{answer_focus} steps"
            ])
        elif answer_intent == 'cost_time':
            search_queries.extend([
                f"{answer_focus} cost",
                f"{answer_focus} fee",
                f"{answer_focus} timeline",
                f"{answer_focus} duration"
            ])
        elif answer_intent == 'legal_specific':
            search_queries.extend([
                f"{answer_focus} clause",
                f"{answer_focus} agreement",
                f"{answer_focus} requirement",
                f"{answer_focus} validity"
            ])
        
        # Perform searches and collect results
        all_results = []
        seen_ids = set()
        
        for i, search_query in enumerate(search_queries):
            try:
                results = search_similar(search_query, top_k)
                
                for doc_id, score, text in results:
                    if doc_id not in seen_ids and score >= VECTOR_SIMILARITY_THRESHOLD:
                        # Boost score for answer-focused content
                        answer_score = self._calculate_answer_score(text, query, answer_intent)
                        boosted_score = score + answer_score
                        
                        all_results.append((doc_id, boosted_score, text, i))
                        seen_ids.add(doc_id)
                        
            except Exception as e:
                logger.warning(f"Search failed for query '{search_query}': {e}")
                continue
        
        # Sort by boosted score and return top results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return [(doc_id, score, text) for doc_id, score, text, _ in all_results[:top_k]]
    
    def _calculate_answer_score(self, text: str, query: str, answer_intent: str) -> float:
        """Calculate additional score for answer-focused content."""
        text_lower = text.lower()
        query_lower = query.lower()
        score_boost = 0.0
        
        # Boost for direct answer indicators
        answer_indicators_found = sum(1 for indicator in self.query_processor.answer_indicators 
                                    if indicator in text_lower)
        score_boost += min(answer_indicators_found * 0.1, 0.3)
        
        # Boost for specific information keywords
        specific_info_found = sum(1 for keyword in self.query_processor.specific_info_keywords 
                                if keyword in text_lower)
        score_boost += min(specific_info_found * 0.05, 0.2)
        
        # Intent-specific boosts
        if answer_intent == 'definition' and any(word in text_lower for word in ['is', 'means', 'refers to', 'defined as']):
            score_boost += 0.15
        elif answer_intent == 'procedure' and any(word in text_lower for word in ['step', 'process', 'procedure', 'method']):
            score_boost += 0.15
        elif answer_intent == 'cost_time' and any(word in text_lower for word in ['cost', 'fee', 'amount', 'time', 'days', 'months']):
            score_boost += 0.15
        elif answer_intent == 'factual' and any(word in text_lower for word in ['yes', 'no', 'true', 'false', 'correct', 'incorrect']):
            score_boost += 0.15
        
        # Boost for direct question-answer patterns
        if re.search(r'question?\s*:.*answer?\s*:', text_lower) or re.search(r'q\s*:.*a\s*:', text_lower):
            score_boost += 0.2
        
        # Boost for numbered/listed information (often contains specific answers)
        if re.search(r'\d+[\.\)]\s+', text) or re.search(r'[•\-\*]\s+', text):
            score_boost += 0.1
        
        return min(score_boost, 0.5)  # Cap the boost
    
    def answer_focused_kg_search(self, query: str, answer_intent: str, top_k: int = 5) -> List[str]:
        """Knowledge graph search focused on finding answers."""
        if not KG_AVAILABLE:
            return []
        
        try:
            # Extract key terms for KG search
            answer_focus = self.query_processor.extract_answer_focus(query)
            
            # Search KG with both original query and answer focus
            kg_results = []
            
            # Primary search
            primary_results = query_kg(query, self.knowledge_graph, include_neighbors=True)
            kg_results.extend(primary_results)
            
            # Focus-based search
            if answer_focus != query:
                focus_results = query_kg(answer_focus, self.knowledge_graph, include_neighbors=True)
                kg_results.extend(focus_results)
            
            # Intent-specific searches
            if answer_intent == 'legal_specific':
                legal_results = query_kg(f"arbitration {answer_focus}", self.knowledge_graph)
                kg_results.extend(legal_results)
            
            # Remove duplicates and return
            unique_results = list(dict.fromkeys(kg_results))
            return unique_results[:top_k]
            
        except Exception as e:
            logger.warning(f"KG search failed: {e}")
            return []
    
    def filter_answer_content(self, sources: List[Dict[str, Any]], query: str, answer_intent: str) -> List[Dict[str, Any]]:
        """Filter and rank sources based on answer relevance."""
        if not sources:
            return sources
        
        answer_focus = self.query_processor.extract_answer_focus(query)
        ranked_sources = []
        
        for source in sources:
            text = source.get('text', '')
            base_score = source.get('score', 0.0)
            
            # Calculate answer relevance
            answer_relevance = self._calculate_answer_score(text, query, answer_intent)
            
            # Check for specific answer patterns
            contains_answer_pattern = self._contains_answer_pattern(text, answer_focus, answer_intent)
            
            # Final ranking score
            final_score = base_score + answer_relevance + (0.2 if contains_answer_pattern else 0)
            
            source_copy = source.copy()
            source_copy['answer_score'] = answer_relevance
            source_copy['final_score'] = final_score
            source_copy['contains_answer'] = contains_answer_pattern
            
            ranked_sources.append(source_copy)
        
        # Sort by final score
        ranked_sources.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_sources
    
    def _contains_answer_pattern(self, text: str, answer_focus: str, answer_intent: str) -> bool:
        """Check if text contains likely answer patterns."""
        text_lower = text.lower()
        focus_lower = answer_focus.lower()
        
        # Check for direct answer patterns
        answer_patterns = [
            f"{focus_lower} is",
            f"{focus_lower} are",
            f"{focus_lower} means",
            f"{focus_lower} requires",
            f"the answer is",
            f"according to",
            f"based on"
        ]
        
        for pattern in answer_patterns:
            if pattern in text_lower:
                return True
        
        # Check for intent-specific patterns
        if answer_intent == 'cost_time':
            cost_patterns = [r'\$[\d,]+', r'\d+\s*(?:days|months|years)', r'\d+%']
            if any(re.search(pattern, text) for pattern in cost_patterns):
                return True
        
        elif answer_intent == 'procedure':
            if re.search(r'\d+[\.\)]\s+', text) or 'step' in text_lower:
                return True
        
        elif answer_intent == 'factual':
            if any(word in text_lower for word in ['yes,', 'no,', 'true', 'false', 'valid', 'invalid']):
                return True
        
        return False
    
    def retrieve_answers(self, query: str, max_context_length: int = 6000) -> Dict[str, Any]:
        """Main retrieval function focused on finding answers."""
        logger.info(f"Answer-focused retrieval for: {query}")
        
        # Classify the answer intent
        answer_intent = self.query_processor.classify_answer_intent(query)
        answer_focus = self.query_processor.extract_answer_focus(query)
        
        logger.info(f"Answer intent: {answer_intent}, Focus: {answer_focus}")
        
        # Perform answer-focused searches
        vector_results = self.answer_focused_vector_search(query, answer_intent, top_k=8)
        kg_results = self.answer_focused_kg_search(query, answer_intent, top_k=5)
        
        # Prepare sources
        all_sources = []
        seen_texts = set()
        
        # Add vector results
        for doc_id, score, text in vector_results:
            if text not in seen_texts:
                all_sources.append({
                    'id': doc_id,
                    'text': text,
                    'score': score,
                    'source_type': 'vector',
                    'metadata': self._get_item_metadata(doc_id)
                })
                seen_texts.add(text)
        
        # Add KG results
        for i, text in enumerate(kg_results):
            if text not in seen_texts:
                all_sources.append({
                    'id': f'kg_answer_{i+1}',
                    'text': text,
                    'score': 0.7 - (i * 0.1),
                    'source_type': 'knowledge_graph',
                    'metadata': {}
                })
                seen_texts.add(text)
        
        # Filter and rank for answer relevance
        filtered_sources = self.filter_answer_content(all_sources, query, answer_intent)
        
        # Build context prioritizing answer-containing sources
        context_parts = []
        current_length = 0
        
        for source in filtered_sources:
            if current_length >= max_context_length:
                break
            
            source_text = source['text']
            if source.get('contains_answer'):
                # Prioritize answer-containing content
                context_parts.insert(0, f"[ANSWER SOURCE] {source_text}")
            else:
                context_parts.append(f"[CONTEXT] {source_text}")
            
            current_length += len(source_text)
        
        return {
            'query_type': answer_intent,
            'answer_focus': answer_focus,
            'context_text': '\n\n'.join(context_parts),
            'sources': filtered_sources,
            'vector_results': vector_results,
            'kg_results': kg_results,
            'total_sources': len(filtered_sources),
            'answer_sources': len([s for s in filtered_sources if s.get('contains_answer')]),
            'search_summary': f"Found {len(filtered_sources)} sources for {answer_intent} about '{answer_focus}'"
        }
    
    def _get_item_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a document item."""
        try:
            manager = get_embedding_manager()
            if hasattr(manager, 'metadata') and manager.metadata:
                for idx, item in manager.metadata.items():
                    if item['id'] == doc_id:
                        return item.get('metadata', {})
            return {}
        except Exception:
            return {}


# Global retriever instance
_retriever = None

def get_retriever() -> AnswerFocusedRetriever:
    """Get the global answer-focused retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = AnswerFocusedRetriever()
    return _retriever

def retrieve_for_query(query: str, **kwargs) -> Dict[str, Any]:
    """Main retrieval function focused on finding answers."""
    retriever = get_retriever()
    return retriever.retrieve_answers(query, **kwargs)

def search_for_answers(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """Quick answer-focused search."""
    retriever = get_retriever()
    answer_intent = retriever.query_processor.classify_answer_intent(query)
    return retriever.answer_focused_vector_search(query, answer_intent, top_k)
