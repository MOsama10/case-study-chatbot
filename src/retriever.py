# """
# Retrieval system combining vector search and knowledge graph queries.
# Handles multi-document collections with advanced query processing.
# """

# from typing import List, Dict, Any, Tuple, Optional, Set
# import logging
# import re
# import numpy as np
# from collections import defaultdict, Counter

# from .config import get_logger, VECTOR_TOP_K, VECTOR_SIMILARITY_THRESHOLD
# from .embeddings import get_embedding_manager, search_similar

# # Knowledge Graph import with enhanced fallback
# try:
#     from .enhanced_knowledge_graph import get_enhanced_knowledge_graph, query_enhanced_kg
#     ENHANCED_KG_AVAILABLE = True
# except ImportError:
#     ENHANCED_KG_AVAILABLE = False

# try:
#     from .knowledge_graph import get_knowledge_graph, query_kg
#     KG_AVAILABLE = True
# except ImportError:
#     # fallback if get_knowledge_graph not available
#     try:
#         from .knowledge_graph import load_kg, query_kg
#         def get_knowledge_graph():
#             return load_kg()
#         KG_AVAILABLE = True
#     except ImportError:
#         # fallback dummy functions
#         import networkx as nx
#         def get_knowledge_graph():
#             return nx.Graph()

#         def query_kg(query, kg=None, include_neighbors=True):
#             return []

#         KG_AVAILABLE = False


# logger = get_logger(__name__)



# class QueryProcessor:
#     """Advanced query processing and analysis."""
    
#     def __init__(self):
#         """Initialize query processor."""
#         self.intent_keywords = {
#             'qa': ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will'],
#             'analysis': ['analyze', 'compare', 'evaluate', 'assess', 'examine', 'study', 'review', 'investigate'],
#             'search': ['find', 'search', 'look for', 'show me', 'list', 'get', 'retrieve', 'display'],
#             'summary': ['summarize', 'overview', 'summary', 'brief', 'outline', 'recap'],
#             'trend': ['trend', 'pattern', 'change', 'evolution', 'development', 'progression'],
#             'recommendation': ['recommend', 'suggest', 'advice', 'should', 'best practice', 'solution']
#         }
        
#         self.domain_keywords = {
#             'customer_service': ['customer', 'service', 'support', 'complaint', 'satisfaction', 'response time'],
#             'hr': ['employee', 'staff', 'turnover', 'hiring', 'training', 'performance', 'benefits'],
#             'sales': ['sales', 'revenue', 'profit', 'customer acquisition', 'conversion', 'pipeline'],
#             'operations': ['process', 'efficiency', 'workflow', 'productivity', 'cost', 'quality'],
#             'technology': ['system', 'software', 'digital', 'automation', 'IT', 'technology'],
#             'finance': ['budget', 'cost', 'expense', 'financial', 'ROI', 'investment', 'profit']
#         }
    
#     def classify_query_intent(self, query: str) -> str:
#         """
#         Classify the intent of the query.
        
#         Args:
#             query: User query
            
#         Returns:
#             Query intent classification
#         """
#         query_lower = query.lower()
        
#         # Check for question patterns
#         if query.endswith('?') or any(word in query_lower for word in self.intent_keywords['qa']):
#             return 'qa'
        
#         # Check for analysis intent
#         if any(word in query_lower for word in self.intent_keywords['analysis']):
#             return 'analysis'
        
#         # Check for search intent
#         if any(word in query_lower for word in self.intent_keywords['search']):
#             return 'search'
        
#         # Check for summary intent
#         if any(word in query_lower for word in self.intent_keywords['summary']):
#             return 'summary'
        
#         # Check for trend analysis
#         if any(word in query_lower for word in self.intent_keywords['trend']):
#             return 'trend'
        
#         # Check for recommendations
#         if any(word in query_lower for word in self.intent_keywords['recommendation']):
#             return 'recommendation'
        
#         return 'general'
    
#     def identify_domain(self, query: str) -> List[str]:
#         """
#         Identify the business domain(s) of the query.
        
#         Args:
#             query: User query
            
#         Returns:
#             List of relevant domains
#         """
#         query_lower = query.lower()
#         relevant_domains = []
        
#         for domain, keywords in self.domain_keywords.items():
#             if any(keyword in query_lower for keyword in keywords):
#                 relevant_domains.append(domain)
        
#         return relevant_domains if relevant_domains else ['general']
    
#     def extract_key_entities(self, query: str) -> List[str]:
#         """
#         Extract key entities and terms from the query.
        
#         Args:
#             query: User query
            
#         Returns:
#             List of key entities
#         """
#         # Remove common stop words and extract meaningful terms
#         stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
#         # Extract words and clean them
#         words = re.findall(r'\b\w+\b', query.lower())
#         entities = [word for word in words if word not in stop_words and len(word) > 2]
        
#         # Extract quoted phrases
#         quoted_phrases = re.findall(r'"([^"]*)"', query)
#         entities.extend(quoted_phrases)
        
#         return entities
    
#     def expand_query(self, query: str, entities: List[str], domains: List[str]) -> List[str]:
#         """
#         Expand query with related terms for better retrieval.
        
#         Args:
#             query: Original query
#             entities: Extracted entities
#             domains: Identified domains
            
#         Returns:
#             List of expanded query terms
#         """
#         expanded_terms = [query]
        
#         # Add domain-specific synonyms
#         domain_synonyms = {
#             'customer_service': ['client support', 'help desk', 'customer care'],
#             'hr': ['human resources', 'personnel', 'workforce'],
#             'sales': ['selling', 'marketing', 'business development'],
#             'operations': ['procedures', 'methods', 'systems'],
#             'technology': ['tech', 'digital solutions', 'IT systems'],
#             'finance': ['financial management', 'accounting', 'monetary']
#         }
        
#         for domain in domains:
#             if domain in domain_synonyms:
#                 expanded_terms.extend(domain_synonyms[domain])
        
#         # Add entity combinations
#         if len(entities) > 1:
#             for i in range(len(entities) - 1):
#                 expanded_terms.append(f"{entities[i]} {entities[i+1]}")
        
#         return expanded_terms


# class DocumentFilter:
#     """Filter and rank documents based on various criteria."""
    
#     def __init__(self):
#         """Initialize document filter."""
#         pass
    
#     def filter_by_document_type(self, sources: List[Dict[str, Any]], doc_types: List[str]) -> List[Dict[str, Any]]:
#         """Filter sources by document type."""
#         if not doc_types:
#             return sources
        
#         filtered = []
#         for source in sources:
#             doc_name = source.get('metadata', {}).get('document_name', '')
#             if any(doc_type.lower() in doc_name.lower() for doc_type in doc_types):
#                 filtered.append(source)
        
#         return filtered if filtered else sources
    
#     def filter_by_recency(self, sources: List[Dict[str, Any]], prefer_recent: bool = True) -> List[Dict[str, Any]]:
#         """Filter/rank sources by document recency (based on filename patterns)."""
#         if not prefer_recent:
#             return sources
        
#         # Try to extract dates or version numbers from filenames
#         dated_sources = []
#         for source in sources:
#             doc_name = source.get('metadata', {}).get('document_name', '')
            
#             # Look for date patterns (YYYY, MM-YYYY, etc.)
#             date_match = re.search(r'20\d{2}', doc_name)
#             version_match = re.search(r'v(\d+)', doc_name, re.IGNORECASE)
            
#             score_boost = 0
#             if date_match:
#                 year = int(date_match.group())
#                 score_boost = (year - 2020) * 0.1  # Boost recent years
#             elif version_match:
#                 version = int(version_match.group(1))
#                 score_boost = version * 0.05  # Boost higher versions
            
#             source_copy = source.copy()
#             source_copy['recency_score'] = source_copy.get('score', 0) + score_boost
#             dated_sources.append(source_copy)
        
#         # Sort by recency score
#         return sorted(dated_sources, key=lambda x: x.get('recency_score', 0), reverse=True)
    
#     def deduplicate_sources(self, sources: List[Dict[str, Any]], similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
#         """Remove duplicate or very similar sources."""
#         if len(sources) <= 1:
#             return sources
        
#         deduplicated = []
#         seen_texts = set()
        
#         for source in sources:
#             text = source['text'].lower().strip()
            
#             # Check for exact duplicates
#             if text in seen_texts:
#                 continue
            
#             # Check for high similarity with existing sources
#             is_duplicate = False
#             for existing_text in seen_texts:
#                 # Simple similarity check based on word overlap
#                 words1 = set(text.split())
#                 words2 = set(existing_text.split())
                
#                 if len(words1) > 0 and len(words2) > 0:
#                     overlap = len(words1.intersection(words2))
#                     similarity = overlap / min(len(words1), len(words2))
                    
#                     if similarity > similarity_threshold:
#                         is_duplicate = True
#                         break
            
#             if not is_duplicate:
#                 deduplicated.append(source)
#                 seen_texts.add(text)
        
#         return deduplicated


# class HybridRetriever:
#     """Advanced retrieval system combining multiple search strategies."""
    
#     def __init__(self):
#         """Initialize retriever."""
#         self.embedding_manager = get_embedding_manager()
        
#         # Safe knowledge graph initialization
#         try:
#             self.knowledge_graph = get_knowledge_graph()
#         except Exception as e:
#             logger.warning(f"Failed to load knowledge graph: {e}")
#             import networkx as nx
#             self.knowledge_graph = nx.Graph()  # Empty graph as fallback
        
#         self.query_processor = QueryProcessor()
#         self.document_filter = DocumentFilter()
        
    
#     def vector_search(self, query: str, top_k: int = VECTOR_TOP_K, filters: Optional[Dict] = None) -> List[Tuple[str, float, str]]:
#         """
#         Perform vector similarity search with optional filtering.
        
#         Args:
#             query: Search query
#             top_k: Number of results to return
#             filters: Optional filters for documents
            
#         Returns:
#             List of (id, score, text) tuples
#         """
#         try:
#             # Get expanded query terms
#             entities = self.query_processor.extract_key_entities(query)
#             domains = self.query_processor.identify_domain(query)
#             expanded_queries = self.query_processor.expand_query(query, entities, domains)
            
#             all_results = []
            
#             # Search with original query
#             results = search_similar(query, top_k * 2)  # Get more results for filtering
#             for doc_id, score, text in results:
#                 if score >= VECTOR_SIMILARITY_THRESHOLD:
#                     all_results.append((doc_id, score, text, 'original'))
            
#             # Search with expanded queries (lower weight)
#             for expanded_query in expanded_queries[1:3]:  # Use top 2 expansions
#                 results = search_similar(expanded_query, top_k)
#                 for doc_id, score, text in results:
#                     if score >= VECTOR_SIMILARITY_THRESHOLD * 0.8:  # Lower threshold for expanded
#                         all_results.append((doc_id, score * 0.7, text, 'expanded'))  # Lower weight
            
#             # Remove duplicates and sort
#             seen_ids = set()
#             unique_results = []
#             for doc_id, score, text, query_type in all_results:
#                 if doc_id not in seen_ids:
#                     unique_results.append((doc_id, score, text))
#                     seen_ids.add(doc_id)
            
#             # Sort by score and return top_k
#             unique_results.sort(key=lambda x: x[1], reverse=True)
#             return unique_results[:top_k]
            
#         except Exception as e:
#             logger.error(f"Vector search error: {e}")
#             return []
    
#     def kg_search(self, query: str, top_k: int = 5, intent: str = 'general') -> List[str]:
#         """
#         Query knowledge graph with intent-aware search.
        
#         Args:
#             query: Search query
#             top_k: Maximum number of results
#             intent: Query intent for specialized search
            
#         Returns:
#             List of relevant text snippets from KG
#         """
#         try:
#             kg_results = query_kg(query, top_k)
            
#             # Intent-specific KG enhancement
#             if intent == 'analysis':
#                 # For analysis queries, also search for related problems and solutions
#                 entities = self.query_processor.extract_key_entities(query)
#                 for entity in entities[:2]:  # Top 2 entities
#                     related_results = query_kg(f"problem solution {entity}", top_k // 2)
#                     kg_results.extend(related_results)
            
#             elif intent == 'recommendation':
#                 # For recommendation queries, focus on solutions
#                 solution_results = query_kg(f"solution approach method {query}", top_k)
#                 kg_results.extend(solution_results)
            
#             # Remove duplicates
#             unique_results = list(dict.fromkeys(kg_results))
#             return unique_results[:top_k]
            
#         except Exception as e:
#             logger.error(f"Knowledge graph search error: {e}")
#             return []
    
#     def keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
#         """
#         Simple keyword-based search as fallback.
        
#         Args:
#             query: Search query
#             top_k: Number of results
            
#         Returns:
#             List of matching items
#         """
#         try:
#             # This is a simplified implementation
#             # In a full system, you might use Elasticsearch or similar
            
#             entities = self.query_processor.extract_key_entities(query)
#             keyword_results = []
            
#             # Search through metadata if available
#             manager = get_embedding_manager()
#             if hasattr(manager, 'metadata') and manager.metadata:
#                 for idx, item in manager.metadata.items():
#                     text_lower = item['text'].lower()
#                     matches = sum(1 for entity in entities if entity in text_lower)
                    
#                     if matches > 0:
#                         score = matches / len(entities) if entities else 0
#                         keyword_results.append({
#                             'id': item['id'],
#                             'text': item['text'],
#                             'score': score,
#                             'source_type': 'keyword'
#                         })
            
#             # Sort by score and return top results
#             keyword_results.sort(key=lambda x: x['score'], reverse=True)
#             return keyword_results[:top_k]
            
#         except Exception as e:
#             logger.error(f"Keyword search error: {e}")
#             return []
    
#     def hybrid_search(self, query: str, vector_k: int = 5, kg_k: int = 3, **kwargs) -> Dict[str, Any]:
#         """
#         Perform hybrid search combining multiple strategies.
        
#         Args:
#             query: Search query
#             vector_k: Number of vector search results
#             kg_k: Number of KG results
#             **kwargs: Additional parameters
            
#         Returns:
#             Dictionary containing search results and metadata
#         """
#         logger.info(f"Performing hybrid search for: {query}")
        
#         # Analyze query
#         intent = self.query_processor.classify_query_intent(query)
#         domains = self.query_processor.identify_domain(query)
#         entities = self.query_processor.extract_key_entities(query)
        
#         # Adjust search parameters based on intent
#         if intent == 'analysis':
#             vector_k, kg_k = min(vector_k + 2, 10), min(kg_k + 2, 8)
#         elif intent == 'search':
#             vector_k, kg_k = min(vector_k + 3, 12), kg_k
#         elif intent == 'summary':
#             vector_k, kg_k = min(vector_k + 1, 8), min(kg_k + 1, 5)
        
#         # Perform searches
#         vector_results = self.vector_search(query, vector_k)
#         kg_results = self.kg_search(query, kg_k, intent)
#         keyword_results = self.keyword_search(query, 3)
        
#         # Combine and process results
#         all_sources = []
#         seen_texts = set()
        
#         # Add vector results
#         for doc_id, score, text in vector_results:
#             if text not in seen_texts:
#                 all_sources.append({
#                     'id': doc_id,
#                     'text': text,
#                     'score': score,
#                     'source_type': 'vector',
#                     'metadata': self._get_item_metadata(doc_id)
#                 })
#                 seen_texts.add(text)
        
#         # Add KG results
#         for i, text in enumerate(kg_results):
#             if text not in seen_texts:
#                 all_sources.append({
#                     'id': f'kg_{i+1}',
#                     'text': text,
#                     'score': 0.8 - (i * 0.1),  # Decreasing score
#                     'source_type': 'knowledge_graph',
#                     'metadata': {}
#                 })
#                 seen_texts.add(text)
        
#         # Add keyword results
#         for result in keyword_results:
#             if result['text'] not in seen_texts:
#                 all_sources.append(result)
#                 seen_texts.add(result['text'])
        
#         # Apply filters and ranking
#         filtered_sources = self.document_filter.deduplicate_sources(all_sources)
        
#         # Apply document type filtering if domains identified
#         if domains and domains != ['general']:
#             filtered_sources = self.document_filter.filter_by_document_type(filtered_sources, domains)
        
#         # Apply recency filtering for certain intents
#         if intent in ['trend', 'summary', 'recommendation']:
#             filtered_sources = self.document_filter.filter_by_recency(filtered_sources, prefer_recent=True)
        
#         # Re-rank combined results
#         final_sources = self._rerank_sources(filtered_sources, query, intent)
        
#         return {
#             'vector_results': vector_results,
#             'kg_results': kg_results,
#             'keyword_results': keyword_results,
#             'combined_sources': final_sources,
#             'query_analysis': {
#                 'intent': intent,
#                 'domains': domains,
#                 'entities': entities,
#                 'expanded_terms': self.query_processor.expand_query(query, entities, domains)
#             },
#             'search_stats': {
#                 'total_vector_results': len(vector_results),
#                 'total_kg_results': len(kg_results),
#                 'total_keyword_results': len(keyword_results),
#                 'final_results': len(final_sources)
#             }
#         }
    
#     def _get_item_metadata(self, doc_id: str) -> Dict[str, Any]:
#         """Get metadata for a document item."""
#         try:
#             manager = get_embedding_manager()
#             if hasattr(manager, 'metadata') and manager.metadata:
#                 for idx, item in manager.metadata.items():
#                     if item['id'] == doc_id:
#                         return item.get('metadata', {})
#             return {}
#         except Exception:
#             return {}
    
#     def _rerank_sources(self, sources: List[Dict[str, Any]], query: str, intent: str) -> List[Dict[str, Any]]:
#         """
#         Re-rank sources based on query intent and other factors.
        
#         Args:
#             sources: List of source documents
#             query: Original query
#             intent: Query intent
            
#         Returns:
#             Re-ranked list of sources
#         """
#         query_lower = query.lower()
        
#         for source in sources:
#             base_score = source.get('score', 0)
#             boost = 0
            
#             # Intent-based boosting
#             text_lower = source['text'].lower()
            
#             if intent == 'analysis':
#                 # Boost sources with analytical terms
#                 if any(term in text_lower for term in ['result', 'outcome', 'impact', 'analysis', 'conclusion']):
#                     boost += 0.1
            
#             elif intent == 'recommendation':
#                 # Boost sources with solution-oriented content
#                 if any(term in text_lower for term in ['solution', 'approach', 'method', 'recommend', 'suggest']):
#                     boost += 0.15
            
#             elif intent == 'trend':
#                 # Boost sources with trend indicators
#                 if any(term in text_lower for term in ['trend', 'increase', 'decrease', 'change', 'over time']):
#                     boost += 0.1
            
#             # Source type boosting
#             if source['source_type'] == 'vector':
#                 boost += 0.05  # Slight preference for vector results
#             elif source['source_type'] == 'knowledge_graph':
#                 boost += 0.03
            
#             # Document quality indicators
#             if len(source['text']) > 200:  # Prefer more substantial content
#                 boost += 0.02
            
#             # Apply boost
#             source['final_score'] = base_score + boost
        
#         # Sort by final score
#         return sorted(sources, key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)
    
#     def retrieve_context(self, query: str, max_context_length: int = 4000, **kwargs) -> Dict[str, Any]:
#         """
#         Retrieve and prepare context for the query with advanced processing.
        
#         Args:
#             query: User query
#             max_context_length: Maximum context length in characters
#             **kwargs: Additional parameters
            
#         Returns:
#             Context information for the agent
#         """
#         # Analyze query first
#         intent = self.query_processor.classify_query_intent(query)
#         domains = self.query_processor.identify_domain(query)
        
#         # Adjust retrieval parameters based on query complexity
#         if intent == 'analysis':
#             vector_k, kg_k = 8, 6
#         elif intent == 'summary':
#             vector_k, kg_k = 10, 4
#         elif intent == 'recommendation':
#             vector_k, kg_k = 6, 8
#         else:
#             vector_k, kg_k = 5, 3
        
#         # Perform hybrid search
#         search_results = self.hybrid_search(query, vector_k=vector_k, kg_k=kg_k, **kwargs)
        
#         # Prepare context with intelligent length management
#         context_parts = []
#         current_length = 0
#         source_documents = defaultdict(list)
        
#         # Group sources by document for better context
#         for source in search_results['combined_sources']:
#             doc_name = source.get('metadata', {}).get('document_name', 'unknown')
#             source_documents[doc_name].append(source)
        
#         # Add context from each document proportionally
#         for doc_name, doc_sources in source_documents.items():
#             doc_context = f"\n--- From {doc_name} ---\n"
            
#             for source in doc_sources[:3]:  # Max 3 sources per document
#                 source_text = f"[{source['source_type'].upper()}] {source['text']}"
                
#                 if current_length + len(doc_context) + len(source_text) <= max_context_length:
#                     if doc_context not in context_parts:
#                         context_parts.append(doc_context)
#                         current_length += len(doc_context)
                    
#                     context_parts.append(source_text)
#                     current_length += len(source_text)
#                 else:
#                     # Truncate last source if needed
#                     remaining_space = max_context_length - current_length
#                     if remaining_space > 100:
#                         truncated_text = source_text[:remaining_space] + "..."
#                         context_parts.append(truncated_text)
#                     break
            
#             if current_length >= max_context_length * 0.9:  # Stop at 90% capacity
#                 break
        
#         # Generate search summary
#         search_summary = self._generate_search_summary(search_results)
        
#         return {
#             'query_type': intent,
#             'domains': domains,
#             'context_text': '\n\n'.join(context_parts),
#             'sources': search_results['combined_sources'],
#             'vector_results': search_results['vector_results'],
#             'kg_results': search_results['kg_results'],
#             'query_analysis': search_results['query_analysis'],
#             'search_stats': search_results['search_stats'],
#             'search_summary': search_summary,
#             'total_sources': len(search_results['combined_sources']),
#             'documents_searched': len(source_documents)
#         }
    
#     def _generate_search_summary(self, search_results: Dict[str, Any]) -> str:
#         """Generate a summary of the search process."""
#         stats = search_results['search_stats']
#         analysis = search_results['query_analysis']
        
#         summary_parts = [
#             f"Found {stats['final_results']} relevant sources",
#             f"Query intent: {analysis['intent']}",
#             f"Domains: {', '.join(analysis['domains'])}"
#         ]
        
#         if stats['total_vector_results'] > 0:
#             summary_parts.append(f"Vector search: {stats['total_vector_results']} matches")
        
#         if stats['total_kg_results'] > 0:
#             summary_parts.append(f"Knowledge graph: {stats['total_kg_results']} connections")
        
#         return " | ".join(summary_parts)


# # Global retriever instance
# _retriever = None

# def get_retriever() -> HybridRetriever:
#     """Get the global retriever instance."""
#     global _retriever
#     if _retriever is None:
#         _retriever = HybridRetriever()
#     return _retriever


# def vector_search(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
#     """Convenience function for vector search."""
#     retriever = get_retriever()
#     return retriever.vector_search(query, top_k)


# def kg_query(query: str, top_k: int = 5) -> List[str]:
#     """Convenience function for KG search."""
#     retriever = get_retriever()
#     return retriever.kg_search(query, top_k)


# def retrieve_for_query(query: str, **kwargs) -> Dict[str, Any]:
#     """Main retrieval function for queries."""
#     retriever = get_retriever()
#     return retriever.retrieve_context(query, **kwargs)


# def search_documents(query: str, document_filters: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
#     """
#     Search documents with optional filtering.
    
#     Args:
#         query: Search query
#         document_filters: List of document names or types to filter by
#         **kwargs: Additional search parameters
        
#     Returns:
#         Search results with filtering applied
#     """
#     retriever = get_retriever()
#     results = retriever.retrieve_context(query, **kwargs)
    
#     if document_filters:
#         # Apply document filtering
#         filtered_sources = []
#         for source in results['sources']:
#             doc_name = source.get('metadata', {}).get('document_name', '')
#             if any(filter_term.lower() in doc_name.lower() for filter_term in document_filters):
#                 filtered_sources.append(source)
        
#         results['sources'] = filtered_sources
#         results['total_sources'] = len(filtered_sources)
    
#     return results


# if __name__ == "__main__":
#     # For testing
#     test_queries = [
#         "What are the main problems in customer service?",
#         "Analyze the effectiveness of employee training programs",
#         "Find examples of cost reduction strategies",
#         "Summarize the trends in employee turnover",
#         "Recommend solutions for improving customer satisfaction"
#     ]
    
#     retriever = get_retriever()
    
#     for query in test_queries:
#         print(f"\n{'='*60}")
#         print(f"Query: {query}")
#         print(f"{'='*60}")
        
#         results = retriever.retrieve_context(query)
        
#         print(f"Intent: {results['query_type']}")
#         print(f"Domains: {', '.join(results['domains'])}")
#         print(f"Total Sources: {results['total_sources']}")
#         print(f"Documents Searched: {results['documents_searched']}")
#         print(f"Search Summary: {results['search_summary']}")
        
#         print("\nTop Sources:")
#         for i, source in enumerate(results['sources'][:3], 1):
#             print(f"{i}. [{source['source_type'].upper()}] {source['text'][:100]}...")
#             print(f"   Score: {source['score']:.3f}")
        
#         print(f"\nContext Preview:")
#         print(results['context_text'][:300] + "..." if len(results['context_text']) > 300 else results['context_text'])

"""
Answer-focused retrieval system for legal documents.
Prioritizes finding actual answers and specific information from documents.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import re
import numpy as np
from collections import defaultdict

from .config import get_logger, VECTOR_TOP_K, VECTOR_SIMILARITY_THRESHOLD
from .embeddings import get_embedding_manager, search_similar

# Knowledge Graph import with fallback
try:
    from .knowledge_graph import get_knowledge_graph, query_kg
    KG_AVAILABLE = True
except ImportError:
    try:
        from .knowledge_graph import load_kg, query_kg
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