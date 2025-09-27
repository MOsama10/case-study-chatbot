"""
Enhanced Knowledge Graph Manager with better relationship extraction.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Set
import networkx as nx
import pickle

from src.config import get_logger, KG_DIR, DATA_ROOT
from src.data_loader import load_multiple_docx

logger = get_logger(__name__)

class EnhancedKGBuilder:
    """Enhanced knowledge graph builder with better entity and relationship extraction."""
    
    def __init__(self):
        self.problem_indicators = [
            'problem:', 'issue:', 'challenge:', 'difficulty:', 'concern:',
            'bottleneck:', 'obstacle:', 'barrier:', 'pain point:', 'failure:'
        ]
        
        self.solution_indicators = [
            'solution:', 'resolution:', 'fix:', 'approach:', 'method:',
            'strategy:', 'implementation:', 'recommendation:', 'best practice:'
        ]
        
        self.result_indicators = [
            'result:', 'outcome:', 'impact:', 'effect:', 'consequence:',
            'improvement:', 'benefit:', 'success:', 'achievement:', 'metric:'
        ]
        
        self.cause_indicators = [
            'cause:', 'reason:', 'due to:', 'because:', 'root cause:',
            'factor:', 'driver:', 'trigger:', 'source:'
        ]

    def extract_entities_and_relationships(self, text: str) -> Dict[str, List[str]]:
        """Extract structured entities and their relationships from text."""
        entities = {
            'problems': [],
            'solutions': [],
            'results': [],
            'causes': []
        }
        
        lines = text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Identify category based on indicators
            if any(indicator in line_lower for indicator in self.problem_indicators):
                current_category = 'problems'
                # Extract the content after the indicator
                for indicator in self.problem_indicators:
                    if indicator in line_lower:
                        content = line[line_lower.find(indicator) + len(indicator):].strip()
                        if content:
                            entities['problems'].append(content)
                        break
                        
            elif any(indicator in line_lower for indicator in self.solution_indicators):
                current_category = 'solutions'
                for indicator in self.solution_indicators:
                    if indicator in line_lower:
                        content = line[line_lower.find(indicator) + len(indicator):].strip()
                        if content:
                            entities['solutions'].append(content)
                        break
                        
            elif any(indicator in line_lower for indicator in self.result_indicators):
                current_category = 'results'
                for indicator in self.result_indicators:
                    if indicator in line_lower:
                        content = line[line_lower.find(indicator) + len(indicator):].strip()
                        if content:
                            entities['results'].append(content)
                        break
                        
            elif any(indicator in line_lower for indicator in self.cause_indicators):
                current_category = 'causes'
                for indicator in self.cause_indicators:
                    if indicator in line_lower:
                        content = line[line_lower.find(indicator) + len(indicator):].strip()
                        if content:
                            entities['causes'].append(content)
                        break
                        
            elif current_category and len(line) > 20:  # Continue previous category
                entities[current_category].append(line)
        
        return entities

    def build_enhanced_kg(self, items: List[Dict[str, Any]]) -> nx.Graph:
        """Build enhanced knowledge graph with better relationship extraction."""
        logger.info("Building enhanced knowledge graph...")
        
        G = nx.Graph()
        
        # Track entity types for better querying
        entity_tracker = {
            'problems': set(),
            'solutions': set(),
            'results': set(),
            'causes': set(),
            'documents': set()
        }
        
        for item in items:
            doc_name = item["metadata"]["source"]
            case_node = f"doc:{doc_name}"
            
            # Add document node
            if case_node not in G:
                G.add_node(case_node, type="document", name=doc_name)
                entity_tracker['documents'].add(case_node)
            
            # Extract structured entities
            entities = self.extract_entities_and_relationships(item["text"])
            
            # Add entity nodes and relationships
            for entity_type, entity_list in entities.items():
                for i, entity_text in enumerate(entity_list):
                    if len(entity_text.strip()) < 10:  # Skip very short entities
                        continue
                    
                    entity_id = f"{entity_type}:{doc_name}:{i}"
                    
                    # Add entity node
                    G.add_node(
                        entity_id,
                        type=entity_type[:-1],  # Remove 's' from plural
                        text=entity_text,
                        document=doc_name,
                        metadata=item["metadata"]
                    )
                    
                    entity_tracker[entity_type].add(entity_id)
                    
                    # Link to document
                    G.add_edge(case_node, entity_id, relation="contains")
            
            # Add original item as content node
            content_node = f"content:{item['id']}"
            G.add_node(
                content_node,
                type=item["type"],
                text=item["text"],
                metadata=item["metadata"]
            )
            G.add_edge(case_node, content_node, relation=item["type"])
        
        # Add cross-document relationships
        self._add_cross_relationships(G, entity_tracker)
        
        logger.info(f"Enhanced KG built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def _add_cross_relationships(self, G: nx.Graph, entity_tracker: Dict[str, Set[str]]):
        """Add relationships between similar entities across documents."""
        
        # Connect similar problems
        problems = list(entity_tracker['problems'])
        for i, prob1 in enumerate(problems):
            for prob2 in problems[i+1:]:
                if self._are_similar_entities(G.nodes[prob1]['text'], G.nodes[prob2]['text']):
                    G.add_edge(prob1, prob2, relation="similar_problem")
        
        # Connect problems to solutions (heuristic matching)
        for problem in entity_tracker['problems']:
            for solution in entity_tracker['solutions']:
                if self._problem_solution_match(G.nodes[problem]['text'], G.nodes[solution]['text']):
                    G.add_edge(problem, solution, relation="addresses")
        
        # Connect solutions to results
        for solution in entity_tracker['solutions']:
            for result in entity_tracker['results']:
                if G.nodes[solution]['document'] == G.nodes[result]['document']:
                    G.add_edge(solution, result, relation="produces")

    def _are_similar_entities(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """Check if two entities are similar based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) > threshold

    def _problem_solution_match(self, problem_text: str, solution_text: str) -> bool:
        """Heuristic to match problems with potential solutions."""
        problem_words = set(problem_text.lower().split())
        solution_words = set(solution_text.lower().split())
        
        # Look for keyword overlap
        overlap = problem_words.intersection(solution_words)
        
        # Check for domain-specific matches
        domain_matches = {
            'customer': ['service', 'support', 'satisfaction', 'experience'],
            'employee': ['training', 'retention', 'engagement', 'benefits'],
            'cost': ['reduction', 'savings', 'efficiency', 'optimization'],
            'quality': ['improvement', 'control', 'assurance', 'standards']
        }
        
        for domain, related_words in domain_matches.items():
            if domain in problem_words:
                if any(word in solution_words for word in related_words):
                    return True
        
        return len(overlap) >= 2  # At least 2 common words


# Global cache for the enhanced knowledge graph
_enhanced_kg: Union[nx.Graph, None] = None

def build_enhanced_kg() -> nx.Graph:
    """Build enhanced knowledge graph from Word documents."""
    global _enhanced_kg
    
    logger.info("Building enhanced knowledge graph from Word documents...")
    
    data_dir = Path(DATA_ROOT) / "batch_1"
    if not data_dir.exists():
        data_dir = Path(DATA_ROOT)
    
    items = load_multiple_docx(data_dir)
    
    if not items:
        logger.warning("No items extracted from %s", data_dir)
        return nx.Graph()
    
    builder = EnhancedKGBuilder()
    _enhanced_kg = builder.build_enhanced_kg(items)
    
    return _enhanced_kg

def query_enhanced_kg(query: str, kg: nx.Graph = None, max_results: int = 10) -> List[Dict[str, Any]]:
    """Enhanced knowledge graph querying with better relevance."""
    if kg is None:
        kg = get_enhanced_knowledge_graph()
    
    query_lower = query.lower()
    matches = []
    
    # Score-based matching
    for node, data in kg.nodes(data=True):
        score = 0
        node_text = str(data.get("text", "")).lower()
        node_type = data.get("type", "")
        
        # Direct text matching
        if query_lower in node_text:
            score += 2
        
        # Word overlap scoring
        query_words = set(query_lower.split())
        node_words = set(node_text.split())
        overlap = query_words.intersection(node_words)
        score += len(overlap) * 0.5
        
        # Type-based boosting
        if "problem" in query_lower and node_type == "problem":
            score += 1
        elif "solution" in query_lower and node_type == "solution":
            score += 1
        elif "result" in query_lower and node_type == "result":
            score += 1
        
        if score > 0:
            match = {
                "id": node,
                "type": node_type,
                "text": data.get("text", ""),
                "score": score,
                "metadata": data.get("metadata", {}),
                "document": data.get("document", "")
            }
            
            # Add neighbor context for important matches
            if score > 1:
                neighbors = []
                for neighbor in kg.neighbors(node):
                    n_data = kg.nodes[neighbor]
                    edge_data = kg.edges[node, neighbor]
                    neighbors.append({
                        "id": neighbor,
                        "type": n_data.get("type", ""),
                        "relation": edge_data.get("relation", ""),
                        "text": n_data.get("text", "")[:100] + "..." if len(n_data.get("text", "")) > 100 else n_data.get("text", "")
                    })
                match["neighbors"] = neighbors[:3]  # Top 3 neighbors
            
            matches.append(match)
    
    # Sort by score and return top results
    matches.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info("Enhanced KG query '%s' matched %d items", query, len(matches))
    return matches[:max_results]

def get_enhanced_knowledge_graph() -> nx.Graph:
    """Get enhanced knowledge graph singleton."""
    global _enhanced_kg
    if _enhanced_kg is None:
        _enhanced_kg = load_enhanced_kg()
    return _enhanced_kg

def load_enhanced_kg(path: Path = None) -> nx.Graph:
    """Load or build enhanced knowledge graph."""
    if path is None:
        path = Path(KG_DIR) / "enhanced_knowledge_graph.pkl"
    
    if path.exists():
        try:
            with open(path, "rb") as f:
                kg = pickle.load(f)
            if isinstance(kg, nx.Graph):
                logger.info("Enhanced knowledge graph loaded from %s", path)
                return kg
        except Exception as e:
            logger.warning("Failed to load enhanced KG: %s", e)
    
    logger.info("Building new enhanced knowledge graph...")
    kg = build_enhanced_kg()
    save_enhanced_kg(kg, path)
    return kg

def save_enhanced_kg(kg: nx.Graph, path: Path = None) -> None:
    """Save enhanced knowledge graph."""
    if path is None:
        path = Path(KG_DIR) / "enhanced_knowledge_graph.pkl"
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(kg, f)
    logger.info("Enhanced knowledge graph saved at %s", path)