"""
Retrieval system combining vector search and knowledge graph queries.
Handles multi-document collections with advanced query processing.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import re
import numpy as np
from collections import defaultdict, Counter

from .embeddings import get_embedding_manager, search_similar
from .knowledge_graph import get_knowledge_graph, query_kg
from .config import get_logger, VECTOR_TOP_K, VECTOR_SIMILARITY_THRESHOLD

logger = get_logger(__name__)


class QueryProcessor:
    """Advanced query processing and analysis."""
    
    def __init__(self):
        """Initialize query processor."""
        self.intent_keywords = {
            'qa': ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will'],
            'analysis': ['analyze', 'compare', 'evaluate', 'assess', 'examine', 'study', 'review', 'investigate'],
            'search': ['find', 'search', 'look for', 'show me', 'list', 'get', 'retrieve', 'display'],
            'summary': ['summarize', 'overview', 'summary', 'brief', 'outline', 'recap'],
            'trend': ['trend', 'pattern', 'change', 'evolution', 'development', 'progression'],
            'recommendation': ['recommend', 'suggest', 'advice', 'should', 'best practice', 'solution']
        }
        
        self.domain_keywords = {
            'customer_service': ['customer', 'service', 'support', 'complaint', 'satisfaction', 'response time'],
            'hr': ['employee', 'staff', 'turnover', 'hiring', 'training', 'performance', 'benefits'],
            'sales': ['sales', 'revenue', 'profit', 'customer acquisition', 'conversion', 'pipeline'],
            'operations': ['process', 'efficiency', 'workflow', 'productivity', 'cost', 'quality'],
            'technology': ['system', 'software', 'digital', 'automation', 'IT', 'technology'],
            'finance': ['budget', 'cost', 'expense', 'financial', 'ROI', 'investment', 'profit']
        }
    
    def classify_query_intent(self, query: str) -> str:
        """
        Classify the intent of the query.
        
        Args:
            query: User query
            
        Returns:
            Query intent classification
        """
        query_lower = query.lower()
        
        # Check for question patterns
        if query.endswith('?') or any(word in query_lower for word in self.intent_keywords['qa']):
            return 'qa'
        
        # Check for analysis intent
        if any(word in query_lower for word in self.intent_keywords['analysis']):
            return 'analysis'
        
        # Check for search intent
        if any(word in query_lower for word in self.intent_keywords['search']):
            return 'search'
        
        # Check for summary intent
        if any(word in query_lower for word in self.intent_keywords['summary']):
            return 'summary'
        
        # Check for trend analysis
        if any(word in query_lower for word in self.intent_keywords['trend']):
            return 'trend'
        
        # Check for recommendations
        if any(word in query_lower for word in self.intent_keywords['recommendation']):
            return 'recommendation'
        
        return 'general'
    
    def identify_domain(self, query: str) -> List[str]:
        """
        Identify the business domain(s) of the query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant domains
        """
        query_lower = query.lower()
        relevant_domains = []
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_domains.append(domain)
        
        return relevant_domains if relevant_domains else ['general']
    
    def extract_key_entities(self, query: str) -> List[str]:
        """
        Extract key entities and terms from the query.
        
        Args:
            query: User query
            
        Returns:
            List of key entities
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words and clean them
        words = re.findall(r'\b\w+\b', query.lower())
        entities = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_phrases)
        
        return entities
    
    def expand_query(self, query: str, entities: List[str], domains: List[str]) -> List[str]:
        """
        Expand query with related terms for better retrieval.
        
        Args:
            query: Original query
            entities: Extracted entities
            domains: Identified domains
            
        Returns:
            List of expanded query terms
        """
        expanded_terms = [query]
        
        # Add domain-specific synonyms
        domain_synonyms = {
            'customer_service': ['client support', 'help desk', 'customer care'],
            'hr': ['human resources', 'personnel', 'workforce'],
            'sales': ['selling', 'marketing', 'business development'],
            'operations': ['procedures', 'methods', 'systems'],
            'technology': ['tech', 'digital solutions', 'IT systems'],
            'finance': ['financial management', 'accounting', 'monetary']
        }
        
        for domain in domains:
            if domain in domain_synonyms:
                expanded_terms.extend(domain_synonyms[domain])
        
        # Add entity combinations
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                expanded_terms.append(f"{entities[i]} {entities[i+1]}")
        
        return expanded_terms


class DocumentFilter:
    """Filter and rank documents based on various criteria."""
    
    def __init__(self):
        """Initialize document filter."""
        pass
    
    def filter_by_document_type(self, sources: List[Dict[str, Any]], doc_types: List[str]) -> List[Dict[str, Any]]:
        """Filter sources by document type."""
        if not doc_types:
            return sources
        
        filtered = []
        for source in sources:
            doc_name = source.get('metadata', {}).get('document_name', '')
            if any(doc_type.lower() in doc_name.lower() for doc_type in doc_types):
                filtered.append(source)
        
        return filtered if filtered else sources
    
    def filter_by_recency(self, sources: List[Dict[str, Any]], prefer_recent: bool = True) -> List[Dict[str, Any]]:
        """Filter/rank sources by document recency (based on filename patterns)."""
        if not prefer_recent:
            return sources
        
        # Try to extract dates or version numbers from filenames
        dated_sources = []
        for source in sources:
            doc_name = source.get('metadata', {}).get('document_name', '')
            
            # Look for date patterns (YYYY, MM-YYYY, etc.)
            date_match = re.search(r'20\d{2}', doc_name)
            version_match = re.search(r'v(\d+)', doc_name, re.IGNORECASE)
            
            score_boost = 0
            if date_match:
                year = int(date_match.group())
                score_boost = (year - 2020) * 0.1  # Boost recent years
            elif version_match:
                version = int(version_match.group(1))
                score_boost = version * 0.05  # Boost higher versions
            
            source_copy = source.copy()
            source_copy['recency_score'] = source_copy.get('score', 0) + score_boost
            dated_sources.append(source_copy)
        
        # Sort by recency score
        return sorted(dated_sources, key=lambda x: x.get('recency_score', 0), reverse=True)
    
    def deduplicate_sources(self, sources: List[Dict[str, Any]], similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar sources."""
        if len(sources) <= 1:
            return sources
        
        deduplicated = []
        seen_texts = set()
        
        for source in sources:
            text = source['text'].lower().strip()
            
            # Check for exact duplicates
            if text in seen_texts:
                continue
            
            # Check for high similarity with existing sources
            is_duplicate = False
            for existing_text in seen_texts:
                # Simple similarity check based on word overlap
                words1 = set(text.split())
                words2 = set(existing_text.split())
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    similarity = overlap / min(len(words1), len(words2))
                    
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(source)
                seen_texts.add(text)
        
        return deduplicated


class HybridRetriever:
    """Advanced retrieval system combining multiple search strategies."""
    
    def __init__(self):
        """Initialize retriever."""
        self.embedding_manager = get_embedding_manager()
        self.knowledge_graph = get_knowledge_graph()
        self.query_processor = QueryProcessor()
        self.document_filter = DocumentFilter()
        
        # Retrieval strategy weights
        self.strategy_weights = {
            'vector_search': 0.6,
            'kg_search': 0.3,
            'keyword_search': 0.1
        }
    
    def vector_search(self, query: str, top_k: int = VECTOR_TOP_K, filters: Optional[Dict] = None) -> List[Tuple[str, float, str]]:
        """
        Perform vector similarity search with optional filtering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters for documents
            
        Returns:
            List of (id, score, text) tuples
        """
        try:
            # Get expanded query terms
            entities = self.query_processor.extract_key_entities(query)
            domains = self.query_processor.identify_domain(query)
            expanded_queries = self.query_processor.expand_query(query, entities, domains)
            
            all_results = []
            
            # Search with original query
            results = search_similar(query, top_k * 2)  # Get more results for filtering
            for doc_id, score, text in results:
                if score >= VECTOR_SIMILARITY_THRESHOLD:
                    all_results.append((doc_id, score, text, 'original'))
            
            # Search with expanded queries (lower weight)
            for expanded_query in expanded_queries[1:3]:  # Use top 2 expansions
                results = search_similar(expanded_query, top_k)
                for doc_id, score, text in results:
                    if score >= VECTOR_SIMILARITY_THRESHOLD * 0.8:  # Lower threshold for expanded
                        all_results.append((doc_id, score * 0.7, text, 'expanded'))  # Lower weight
            
            # Remove duplicates and sort
            seen_ids = set()
            unique_results = []
            for doc_id, score, text, query_type in all_results:
                if doc_id not in seen_ids:
                    unique_results.append((doc_id, score, text))
                    seen_ids.add(doc_id)
            
            # Sort by score and return top_k
            unique_results.sort(key=lambda x: x[1], reverse=True)
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def kg_search(self, query: str, top_k: int = 5, intent: str = 'general') -> List[str]:
        """
        Query knowledge graph with intent-aware search.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            intent: Query intent for specialized search
            
        Returns:
            List of relevant text snippets from KG
        """
        try:
            kg_results = query_kg(query, top_k)
            
            # Intent-specific KG enhancement
            if intent == 'analysis':
                # For analysis queries, also search for related problems and solutions
                entities = self.query_processor.extract_key_entities(query)
                for entity in entities[:2]:  # Top 2 entities
                    related_results = query_kg(f"problem solution {entity}", top_k // 2)
                    kg_results.extend(related_results)
            
            elif intent == 'recommendation':
                # For recommendation queries, focus on solutions
                solution_results = query_kg(f"solution approach method {query}", top_k)
                kg_results.extend(solution_results)
            
            # Remove duplicates
            unique_results = list(dict.fromkeys(kg_results))
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"Knowledge graph search error: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search as fallback.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching items
        """
        try:
            # This is a simplified implementation
            # In a full system, you might use Elasticsearch or similar
            
            entities = self.query_processor.extract_key_entities(query)
            keyword_results = []
            
            # Search through metadata if available
            manager = get_embedding_manager()
            if hasattr(manager, 'metadata') and manager.metadata:
                for idx, item in manager.metadata.items():
                    text_lower = item['text'].lower()
                    matches = sum(1 for entity in entities if entity in text_lower)
                    
                    if matches > 0:
                        score = matches / len(entities) if entities else 0
                        keyword_results.append({
                            'id': item['id'],
                            'text': item['text'],
                            'score': score,
                            'source_type': 'keyword'
                        })
            
            # Sort by score and return top results
            keyword_results.sort(key=lambda x: x['score'], reverse=True)
            return keyword_results[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def hybrid_search(self, query: str, vector_k: int = 5, kg_k: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Perform hybrid search combining multiple strategies.
        
        Args:
            query: Search query
            vector_k: Number of vector search results
            kg_k: Number of KG results
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing search results and metadata
        """
        logger.info(f"Performing hybrid search for: {query}")
        
        # Analyze query
        intent = self.query_processor.classify_query_intent(query)
        domains = self.query_processor.identify_domain(query)
        entities = self.query_processor.extract_key_entities(query)
        
        # Adjust search parameters based on intent
        if intent == 'analysis':
            vector_k, kg_k = min(vector_k + 2, 10), min(kg_k + 2, 8)
        elif intent == 'search':
            vector_k, kg_k = min(vector_k + 3, 12), kg_k
        elif intent == 'summary':
            vector_k, kg_k = min(vector_k + 1, 8), min(kg_k + 1, 5)
        
        # Perform searches
        vector_results = self.vector_search(query, vector_k)
        kg_results = self.kg_search(query, kg_k, intent)
        keyword_results = self.keyword_search(query, 3)
        
        # Combine and process results
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
                    'id': f'kg_{i+1}',
                    'text': text,
                    'score': 0.8 - (i * 0.1),  # Decreasing score
                    'source_type': 'knowledge_graph',
                    'metadata': {}
                })
                seen_texts.add(text)
        
        # Add keyword results
        for result in keyword_results:
            if result['text'] not in seen_texts:
                all_sources.append(result)
                seen_texts.add(result['text'])
        
        # Apply filters and ranking
        filtered_sources = self.document_filter.deduplicate_sources(all_sources)
        
        # Apply document type filtering if domains identified
        if domains and domains != ['general']:
            filtered_sources = self.document_filter.filter_by_document_type(filtered_sources, domains)
        
        # Apply recency filtering for certain intents
        if intent in ['trend', 'summary', 'recommendation']:
            filtered_sources = self.document_filter.filter_by_recency(filtered_sources, prefer_recent=True)
        
        # Re-rank combined results
        final_sources = self._rerank_sources(filtered_sources, query, intent)
        
        return {
            'vector_results': vector_results,
            'kg_results': kg_results,
            'keyword_results': keyword_results,
            'combined_sources': final_sources,
            'query_analysis': {
                'intent': intent,
                'domains': domains,
                'entities': entities,
                'expanded_terms': self.query_processor.expand_query(query, entities, domains)
            },
            'search_stats': {
                'total_vector_results': len(vector_results),
                'total_kg_results': len(kg_results),
                'total_keyword_results': len(keyword_results),
                'final_results': len(final_sources)
            }
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
    
    def _rerank_sources(self, sources: List[Dict[str, Any]], query: str, intent: str) -> List[Dict[str, Any]]:
        """
        Re-rank sources based on query intent and other factors.
        
        Args:
            sources: List of source documents
            query: Original query
            intent: Query intent
            
        Returns:
            Re-ranked list of sources
        """
        query_lower = query.lower()
        
        for source in sources:
            base_score = source.get('score', 0)
            boost = 0
            
            # Intent-based boosting
            text_lower = source['text'].lower()
            
            if intent == 'analysis':
                # Boost sources with analytical terms
                if any(term in text_lower for term in ['result', 'outcome', 'impact', 'analysis', 'conclusion']):
                    boost += 0.1
            
            elif intent == 'recommendation':
                # Boost sources with solution-oriented content
                if any(term in text_lower for term in ['solution', 'approach', 'method', 'recommend', 'suggest']):
                    boost += 0.15
            
            elif intent == 'trend':
                # Boost sources with trend indicators
                if any(term in text_lower for term in ['trend', 'increase', 'decrease', 'change', 'over time']):
                    boost += 0.1
            
            # Source type boosting
            if source['source_type'] == 'vector':
                boost += 0.05  # Slight preference for vector results
            elif source['source_type'] == 'knowledge_graph':
                boost += 0.03
            
            # Document quality indicators
            if len(source['text']) > 200:  # Prefer more substantial content
                boost += 0.02
            
            # Apply boost
            source['final_score'] = base_score + boost
        
        # Sort by final score
        return sorted(sources, key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)
    
    def retrieve_context(self, query: str, max_context_length: int = 4000, **kwargs) -> Dict[str, Any]:
        """
        Retrieve and prepare context for the query with advanced processing.
        
        Args:
            query: User query
            max_context_length: Maximum context length in characters
            **kwargs: Additional parameters
            
        Returns:
            Context information for the agent
        """
        # Analyze query first
        intent = self.query_processor.classify_query_intent(query)
        domains = self.query_processor.identify_domain(query)
        
        # Adjust retrieval parameters based on query complexity
        if intent == 'analysis':
            vector_k, kg_k = 8, 6
        elif intent == 'summary':
            vector_k, kg_k = 10, 4
        elif intent == 'recommendation':
            vector_k, kg_k = 6, 8
        else:
            vector_k, kg_k = 5, 3
        
        # Perform hybrid search
        search_results = self.hybrid_search(query, vector_k=vector_k, kg_k=kg_k, **kwargs)
        
        # Prepare context with intelligent length management
        context_parts = []
        current_length = 0
        source_documents = defaultdict(list)
        
        # Group sources by document for better context
        for source in search_results['combined_sources']:
            doc_name = source.get('metadata', {}).get('document_name', 'unknown')
            source_documents[doc_name].append(source)
        
        # Add context from each document proportionally
        for doc_name, doc_sources in source_documents.items():
            doc_context = f"\n--- From {doc_name} ---\n"
            
            for source in doc_sources[:3]:  # Max 3 sources per document
                source_text = f"[{source['source_type'].upper()}] {source['text']}"
                
                if current_length + len(doc_context) + len(source_text) <= max_context_length:
                    if doc_context not in context_parts:
                        context_parts.append(doc_context)
                        current_length += len(doc_context)
                    
                    context_parts.append(source_text)
                    current_length += len(source_text)
                else:
                    # Truncate last source if needed
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:
                        truncated_text = source_text[:remaining_space] + "..."
                        context_parts.append(truncated_text)
                    break
            
            if current_length >= max_context_length * 0.9:  # Stop at 90% capacity
                break
        
        # Generate search summary
        search_summary = self._generate_search_summary(search_results)
        
        return {
            'query_type': intent,
            'domains': domains,
            'context_text': '\n\n'.join(context_parts),
            'sources': search_results['combined_sources'],
            'vector_results': search_results['vector_results'],
            'kg_results': search_results['kg_results'],
            'query_analysis': search_results['query_analysis'],
            'search_stats': search_results['search_stats'],
            'search_summary': search_summary,
            'total_sources': len(search_results['combined_sources']),
            'documents_searched': len(source_documents)
        }
    
    def _generate_search_summary(self, search_results: Dict[str, Any]) -> str:
        """Generate a summary of the search process."""
        stats = search_results['search_stats']
        analysis = search_results['query_analysis']
        
        summary_parts = [
            f"Found {stats['final_results']} relevant sources",
            f"Query intent: {analysis['intent']}",
            f"Domains: {', '.join(analysis['domains'])}"
        ]
        
        if stats['total_vector_results'] > 0:
            summary_parts.append(f"Vector search: {stats['total_vector_results']} matches")
        
        if stats['total_kg_results'] > 0:
            summary_parts.append(f"Knowledge graph: {stats['total_kg_results']} connections")
        
        return " | ".join(summary_parts)


# Global retriever instance
_retriever = None

def get_retriever() -> HybridRetriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def vector_search(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """Convenience function for vector search."""
    retriever = get_retriever()
    return retriever.vector_search(query, top_k)


def kg_query(query: str, top_k: int = 5) -> List[str]:
    """Convenience function for KG search."""
    retriever = get_retriever()
    return retriever.kg_search(query, top_k)


def retrieve_for_query(query: str, **kwargs) -> Dict[str, Any]:
    """Main retrieval function for queries."""
    retriever = get_retriever()
    return retriever.retrieve_context(query, **kwargs)


def search_documents(query: str, document_filters: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """
    Search documents with optional filtering.
    
    Args:
        query: Search query
        document_filters: List of document names or types to filter by
        **kwargs: Additional search parameters
        
    Returns:
        Search results with filtering applied
    """
    retriever = get_retriever()
    results = retriever.retrieve_context(query, **kwargs)
    
    if document_filters:
        # Apply document filtering
        filtered_sources = []
        for source in results['sources']:
            doc_name = source.get('metadata', {}).get('document_name', '')
            if any(filter_term.lower() in doc_name.lower() for filter_term in document_filters):
                filtered_sources.append(source)
        
        results['sources'] = filtered_sources
        results['total_sources'] = len(filtered_sources)
    
    return results


if __name__ == "__main__":
    # For testing
    test_queries = [
        "What are the main problems in customer service?",
        "Analyze the effectiveness of employee training programs",
        "Find examples of cost reduction strategies",
        "Summarize the trends in employee turnover",
        "Recommend solutions for improving customer satisfaction"
    ]
    
    retriever = get_retriever()
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve_context(query)
        
        print(f"Intent: {results['query_type']}")
        print(f"Domains: {', '.join(results['domains'])}")
        print(f"Total Sources: {results['total_sources']}")
        print(f"Documents Searched: {results['documents_searched']}")
        print(f"Search Summary: {results['search_summary']}")
        
        print("\nTop Sources:")
        for i, source in enumerate(results['sources'][:3], 1):
            print(f"{i}. [{source['source_type'].upper()}] {source['text'][:100]}...")
            print(f"   Score: {source['score']:.3f}")
        
        print(f"\nContext Preview:")
        print(results['context_text'][:300] + "..." if len(results['context_text']) > 300 else results['context_text'])