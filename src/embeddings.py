"""
Embedding generation and vector index management.
Uses sentence-transformers with FAISS for efficient similarity search.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from pathlib import Path
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

from .config import (
    get_logger, INDEXES_DIR, META_DIR, EMBEDDING_MODEL, 
    EMBEDDING_BATCH_SIZE, MAX_EMBEDDING_RETRIES
)

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages embeddings and vector search operations."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize embedding manager with specified model."""
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = {}
        self.dimension = None
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            NumPy array of embeddings
        """
        self.load_model()
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            
            # Retry logic for embedding generation
            for attempt in range(MAX_EMBEDDING_RETRIES):
                try:
                    batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    if attempt == MAX_EMBEDDING_RETRIES - 1:
                        raise
        
        return np.array(embeddings)
    
    def build_index(self, items: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from document items.
        
        Args:
            items: List of document items with 'text' field
        """
        logger.info("Building embedding index...")
        
        # Extract texts and create metadata mapping
        texts = []
        metadata = {}
        
        for i, item in enumerate(items):
            texts.append(item['text'])
            metadata[i] = {
                'id': item['id'],
                'text': item['text'][:200] + "..." if len(item['text']) > 200 else item['text'],
                'type': item.get('type', 'unknown'),
                'metadata': item.get('metadata', {})
            }
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.metadata = metadata
        logger.info(f"Built index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of (id, score, text) tuples
        """
        if self.index is None or self.model is None:
            raise ValueError("Index not built or model not loaded")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.metadata:
                item = self.metadata[idx]
                results.append((item['id'], float(score), item['text']))
        
        return results
    
    def save_index(self, index_path: Path, metadata_path: Path) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        logger.info(f"Saving index to {index_path}")
        
        # Ensure directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.metadata,
                'model_name': self.model_name,
                'dimension': self.dimension,
                'total_vectors': self.index.ntotal
            }, f, indent=2, ensure_ascii=False)
        
        logger.info("Index and metadata saved successfully")
    
    def load_index(self, index_path: Path, metadata_path: Path) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not index_path.exists() or not metadata_path.exists():
                logger.warning("Index files not found")
                return False
            
            logger.info(f"Loading index from {index_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data['metadata']
                self.model_name = data['model_name']
                self.dimension = data['dimension']
            
            # Load model
            self.load_model()
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False


# Global embedding manager instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def build_embeddings(items: List[Dict[str, Any]]) -> EmbeddingManager:
    """
    Build embeddings for document items.
    
    Args:
        items: List of document items
        
    Returns:
        Configured EmbeddingManager
    """
    manager = get_embedding_manager()
    manager.build_index(items)
    return manager


def save_index(index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
    """Save the current index to disk."""
    if index_path is None:
        index_path = INDEXES_DIR / "faiss_index.bin"
    if metadata_path is None:
        metadata_path = META_DIR / "embeddings_metadata.json"
    
    manager = get_embedding_manager()
    manager.save_index(index_path, metadata_path)


def load_index(index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> bool:
    """Load index from disk."""
    if index_path is None:
        index_path = INDEXES_DIR / "faiss_index.bin"
    if metadata_path is None:
        metadata_path = META_DIR / "embeddings_metadata.json"
    
    manager = get_embedding_manager()
    return manager.load_index(index_path, metadata_path)


def search_similar(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """Search for similar documents."""
    manager = get_embedding_manager()
    return manager.search(query, top_k)


def build_and_save_embeddings(doc_path: Optional[Path] = None) -> None:
    """
    Complete pipeline: load documents, build embeddings, and save.
    
    Args:
        doc_path: Path to document file. If None, uses default data/cases.docx
    """
    from .data_loader import load_docx, chunk_text, preprocess_text
    from .config import DATA_ROOT
    
    if doc_path is None:
        doc_path = DATA_ROOT / "cases.docx"
    
    if not doc_path.exists():
        logger.warning(f"Document not found: {doc_path}")
        logger.info("Creating sample document structure...")
        # Create sample items for development
        items = [
            {
                "id": "sample_1",
                "text": "Sample case study: A company faced declining sales due to poor customer service. The solution was to implement a comprehensive training program.",
                "type": "paragraph",
                "metadata": {"source": "sample", "element_type": "paragraph"}
            },
            {
                "id": "sample_2", 
                "text": "Problem: High employee turnover. Solution: Improved benefits package and remote work options resulted in 40% reduction in turnover.",
                "type": "paragraph",
                "metadata": {"source": "sample", "element_type": "paragraph"}
            }
        ]
    else:
        # Load actual document
        items = load_docx(doc_path)
        
        # Chunk large texts
        chunked_items = []
        for item in items:
            if len(item['text']) > 1000:
                chunks = chunk_text(item['text'])
                for i, chunk in enumerate(chunks):
                    chunked_item = item.copy()
                    chunked_item['id'] = f"{item['id']}_chunk_{i+1}"
                    chunked_item['text'] = preprocess_text(chunk)
                    chunked_items.append(chunked_item)
            else:
                item['text'] = preprocess_text(item['text'])
                chunked_items.append(item)
        
        items = chunked_items
    
    # Build embeddings
    logger.info(f"Building embeddings for {len(items)} items")
    build_embeddings(items)
    
    # Save to disk
    save_index()
    
    logger.info("Embeddings built and saved successfully")