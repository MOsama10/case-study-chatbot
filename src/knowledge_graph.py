"""
Knowledge Graph Manager for the Case Study Chatbot.
Builds the KG from Word documents using the docx loader.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import networkx as nx
import pickle

from src.config import get_logger, KG_DIR, DATA_ROOT
from src.data_loader import load_multiple_docx

logger = get_logger(__name__)

# Global cache for the knowledge graph
_kg: Union[nx.Graph, None] = None


# -----------------------------
# Core KG Functions
# -----------------------------

def build_kg() -> nx.Graph:
    """
    Build a knowledge graph from Word documents in DATA_ROOT.
    Each document = case study node.
    Each paragraph/table row = entity node connected to its document.
    """
    logger.info("Building knowledge graph from Word documents...")

    # Try batch_1 first, then fallback to root data directory
    data_dirs = [Path(DATA_ROOT) / "batch_1", Path(DATA_ROOT)]
    items = []
    
    for data_dir in data_dirs:
        if data_dir.exists():
            try:
                items = load_multiple_docx(data_dir)
                if items:
                    logger.info(f"Loaded {len(items)} items from {data_dir}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load from {data_dir}: {e}")
                continue

    if not items:
        logger.warning("No items extracted from any data directory")
        # Create a minimal sample graph for testing
        return create_sample_kg()

    G = nx.Graph()

    for item in items:
        doc_name = item["metadata"].get("source", "unknown")
        case_node = f"doc:{doc_name}"

        # Ensure case study node exists
        if case_node not in G:
            G.add_node(case_node, type="case", name=doc_name)

        # Create a node for the paragraph/table row
        entity_node = item["id"]
        G.add_node(
            entity_node,
            type=item["type"],
            text=item["text"],
            metadata=item["metadata"]
        )

        # Link entity to its case study document
        G.add_edge(case_node, entity_node, relation=item["type"])

    logger.info("Built KG with %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def create_sample_kg() -> nx.Graph:
    """Create a sample knowledge graph for testing when no documents are available."""
    logger.info("Creating sample knowledge graph...")
    
    G = nx.Graph()
    
    # Sample case study nodes
    case_nodes = [
        ("doc:sample_case_1.docx", {"type": "case", "name": "Customer Service Improvement"}),
        ("doc:sample_case_2.docx", {"type": "case", "name": "Employee Retention Strategy"}),
    ]
    
    # Sample content nodes
    content_nodes = [
        ("para_1", {
            "type": "paragraph", 
            "text": "Problem: Customer service response times were too slow, averaging 48 hours instead of target 4 hours.",
            "metadata": {"source": "sample_case_1.docx"}
        }),
        ("para_2", {
            "type": "paragraph",
            "text": "Solution: Implemented automated ticket routing and hired additional support staff.",
            "metadata": {"source": "sample_case_1.docx"}
        }),
        ("para_3", {
            "type": "paragraph",
            "text": "Problem: High employee turnover rate of 45% annually due to limited career development.",
            "metadata": {"source": "sample_case_2.docx"}
        }),
        ("para_4", {
            "type": "paragraph",
            "text": "Solution: Created mentorship program and skills training, reducing turnover to 15%.",
            "metadata": {"source": "sample_case_2.docx"}
        })
    ]
    
    # Add nodes
    for node_id, data in case_nodes + content_nodes:
        G.add_node(node_id, **data)
    
    # Add edges
    G.add_edge("doc:sample_case_1.docx", "para_1", relation="paragraph")
    G.add_edge("doc:sample_case_1.docx", "para_2", relation="paragraph")
    G.add_edge("doc:sample_case_2.docx", "para_3", relation="paragraph")
    G.add_edge("doc:sample_case_2.docx", "para_4", relation="paragraph")
    
    logger.info("Created sample KG with %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def save_kg(kg: nx.Graph, path: Path = None) -> None:
    """Save knowledge graph to disk as pickle."""
    if path is None:
        path = Path(KG_DIR) / "knowledge_graph.pkl"
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(kg, f)
        logger.info("Knowledge graph saved at %s", path)
    except Exception as e:
        logger.error("Failed to save knowledge graph: %s", e)


def load_kg(path: Path = None) -> nx.Graph:
    """Load knowledge graph from disk or build it if missing/corrupt."""
    global _kg
    if path is None:
        path = Path(KG_DIR) / "knowledge_graph.pkl"

    if not path.exists():
        logger.warning("KG file not found at %s. Rebuilding...", path)
        _kg = build_kg()
        save_kg(_kg, path)
    else:
        try:
            with open(path, "rb") as f:
                _kg = pickle.load(f)
            
            # Validate loaded KG
            if not isinstance(_kg, nx.Graph):
                logger.error("Invalid KG file detected at %s. Rebuilding...", path)
                _kg = build_kg()
                save_kg(_kg, path)
            else:
                logger.info("Knowledge graph loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load KG from %s: %s. Rebuilding...", path, e)
            _kg = build_kg()
            save_kg(_kg, path)

    return _kg


def get_knowledge_graph() -> nx.Graph:
    """Return a singleton KG instance (build/load if needed)."""
    global _kg
    if _kg is None:
        _kg = load_kg()
    return _kg


def query_kg(query: str, kg: nx.Graph = None, include_neighbors: bool = True) -> List[str]:
    """
    Search nodes/edges in the KG relevant to the query.
    Returns a list of text strings from matching nodes.
    """
    if kg is None:
        kg = get_knowledge_graph()

    query_lower = query.lower()
    matches: List[str] = []

    # Search nodes
    for node, data in kg.nodes(data=True):
        node_text = str(data.get("text", ""))
        node_name = str(node).lower()
        
        # Check if query matches node name or text content
        if (query_lower in node_name or 
            (node_text and query_lower in node_text.lower())):
            
            # Add the text content
            if node_text:
                matches.append(node_text)
            
            # Add neighbor context if requested
            if include_neighbors:
                for neighbor in kg.neighbors(node):
                    neighbor_data = kg.nodes[neighbor]
                    neighbor_text = neighbor_data.get("text", "")
                    if neighbor_text and neighbor_text not in matches:
                        matches.append(neighbor_text)

    # Search edge relations
    for u, v, data in kg.edges(data=True):
        relation = data.get("relation", "")
        if query_lower in relation.lower():
            # Add text from connected nodes
            for node_id in [u, v]:
                node_data = kg.nodes[node_id]
                node_text = node_data.get("text", "")
                if node_text and node_text not in matches:
                    matches.append(node_text)

    logger.info("KG query '%s' matched %d items", query, len(matches))
    return matches[:20]  # Limit to top 20 results


def get_kg_stats() -> Dict[str, Any]:
    """Get statistics about the knowledge graph."""
    kg = get_knowledge_graph()
    
    return {
        "nodes": kg.number_of_nodes(),
        "edges": kg.number_of_edges(),
        "node_types": list(set(data.get("type", "unknown") for _, data in kg.nodes(data=True))),
        "edge_relations": list(set(data.get("relation", "unknown") for _, _, data in kg.edges(data=True)))
    }


def rebuild_kg() -> nx.Graph:
    """Force rebuild the knowledge graph."""
    global _kg
    logger.info("Force rebuilding knowledge graph...")
    _kg = build_kg()
    save_kg(_kg)
    return _kg


# -----------------------------
# Debugging and Testing
# -----------------------------
if __name__ == "__main__":
    # Test the knowledge graph functionality
    print("Testing Knowledge Graph...")
    
    kg = get_knowledge_graph()
    stats = get_kg_stats()
    
    print(f"KG Stats: {stats}")
    print(f"Sample nodes: {list(kg.nodes())[:5]}")
    print(f"Sample edges: {list(kg.edges())[:5]}")
    
    # Test queries
    test_queries = ["problem", "solution", "customer", "employee"]
    for query in test_queries:
        results = query_kg(query, kg)
        print(f"Query '{query}': {len(results)} results")
        if results:
            print(f"  First result: {results[0][:100]}...")