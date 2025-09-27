"""
Knowledge Graph Manager for the Case Study Chatbot.
Builds the KG from Word documents in batch_1/ using the docx loader.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import networkx as nx
import pickle

from .config import get_logger, KG_DIR, DATA_ROOT
from .data_loader import load_multiple_docx  # ✅ import your loader

logger = get_logger(__name__)

# Global cache for the knowledge graph
_kg: Union[nx.Graph, None] = None


# -----------------------------
# Core KG Functions
# -----------------------------

def build_kg() -> nx.Graph:
    """
    Build a knowledge graph from Word documents in DATA_ROOT (e.g., batch_1/).
    Each document = case study node.
    Each paragraph/table row = entity node connected to its document.
    """
    logger.info("Building knowledge graph from Word documents...")

    data_dir = Path(DATA_ROOT) / "batch_1"
    items = load_multiple_docx(data_dir)

    if not items:
        logger.warning("No items extracted from %s", data_dir)
        return nx.Graph()

    G = nx.Graph()

    for item in items:
        doc_name = item["metadata"]["source"]
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


def save_kg(kg: nx.Graph, path: Path = None) -> None:
    """Save knowledge graph to disk as pickle."""
    if path is None:
        path = Path(KG_DIR) / "knowledge_graph.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(kg, f)
    logger.info("Knowledge graph saved at %s", path)


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
        with open(path, "rb") as f:
            _kg = pickle.load(f)
        # ✅ safeguard in case old/corrupted pickle was an int or bad type
        if not isinstance(_kg, nx.Graph):
            logger.error("Invalid KG file detected at %s. Rebuilding...", path)
            _kg = build_kg()
            save_kg(_kg, path)
        else:
            logger.info("Knowledge graph loaded from %s", path)

    return _kg


def get_knowledge_graph() -> nx.Graph:
    """Return a singleton KG instance (build/load if needed)."""
    global _kg
    if _kg is None:
        _kg = load_kg()
    return _kg


def query_kg(query: str, kg: nx.Graph = None, include_neighbors: bool = True) -> List[Dict[str, Any]]:
    """
    Search nodes/edges in the KG relevant to the query.
    Optionally expand with neighbor context.
    """
    if kg is None:
        kg = get_knowledge_graph()

    query_lower = query.lower()
    matches: List[Dict[str, Any]] = []

    # Search nodes
    for node, data in kg.nodes(data=True):
        if query_lower in str(node).lower() or query_lower in str(data.get("text", "")).lower():
            match = {
                "id": node,
                "type": data.get("type", "unknown"),
                "text": data.get("text", ""),
                "metadata": data.get("metadata", {})
            }

            # ✅ Include neighbors for context (e.g., Q → A, doc → entities)
            if include_neighbors:
                neighbors = []
                for neighbor in kg.neighbors(node):
                    n_data = kg.nodes[neighbor]
                    neighbors.append({
                        "id": neighbor,
                        "type": n_data.get("type", "unknown"),
                        "text": n_data.get("text", ""),
                        "metadata": n_data.get("metadata", {})
                    })
                match["neighbors"] = neighbors

            matches.append(match)

    # Search relations (edges)
    for u, v, data in kg.edges(data=True):
        if query_lower in str(data.get("relation", "")).lower():
            matches.append({
                "id": f"{u} - {v}",
                "type": "relation",
                "relation": data.get("relation", "")
            })

    logger.info("KG query '%s' matched %d items", query, len(matches))
    return matches


# -----------------------------
# Debugging
# -----------------------------
if __name__ == "__main__":
    kg = get_knowledge_graph()
    print("Nodes:", list(kg.nodes(data=True))[:10])  # first 10 nodes
    print("Edges:", list(kg.edges(data=True))[:10])  # first 10 edges
    print("\nQuery results for 'Q:':")
    print(query_kg("Q:")[:3])  # first 3 matches
