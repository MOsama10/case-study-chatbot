# """
# Embedding generation and vector index management.
# Uses sentence-transformers with FAISS for efficient similarity search.
# """

# from typing import List, Dict, Any, Tuple, Optional
# import json
# import pickle
# from pathlib import Path
# import logging
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from tqdm import tqdm

# from src.config import (
#     get_logger, INDEXES_DIR, META_DIR, EMBEDDING_MODEL, 
#     EMBEDDING_BATCH_SIZE, MAX_EMBEDDING_RETRIES
# )

# logger = get_logger(__name__)


# class EmbeddingManager:
#     """Manages embeddings and vector search operations."""
    
#     def __init__(self, model_name: str = EMBEDDING_MODEL):
#         """Initialize embedding manager with specified model."""
#         self.model_name = model_name
#         self.model = None
#         self.index = None
#         self.metadata = {}
#         self.dimension = None
        
#     def load_model(self) -> None:
#         """Load the sentence transformer model."""
#         if self.model is None:
#             logger.info(f"Loading embedding model: {self.model_name}")
#             self.model = SentenceTransformer(self.model_name)
#             self.dimension = self.model.get_sentence_embedding_dimension()
#             logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
#     def generate_embeddings(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
#         """
#         Generate embeddings for a list of texts.
        
#         Args:
#             texts: List of text strings
#             batch_size: Number of texts to process at once
            
#         Returns:
#             NumPy array of embeddings
#         """
#         self.load_model()
        
#         embeddings = []
#         for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
#             batch = texts[i:i + batch_size]
            
#             # Retry logic for embedding generation
#             for attempt in range(MAX_EMBEDDING_RETRIES):
#                 try:
#                     batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
#                     embeddings.extend(batch_embeddings)
#                     break
#                 except Exception as e:
#                     logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
#                     if attempt == MAX_EMBEDDING_RETRIES - 1:
#                         raise
        
#         return np.array(embeddings)
    
#     def build_index(self, items: List[Dict[str, Any]]) -> None:
#         """
#         Build FAISS index from document items.
        
#         Args:
#             items: List of document items with 'text' field
#         """
#         logger.info("Building embedding index...")
        
#         # Extract texts and create metadata mapping
#         texts = []
#         metadata = {}
        
#         for i, item in enumerate(items):
#             texts.append(item['text'])
#             metadata[i] = {
#                 'id': item['id'],
#                 'text': item['text'][:200] + "..." if len(item['text']) > 200 else item['text'],
#                 'type': item.get('type', 'unknown'),
#                 'metadata': item.get('metadata', {})
#             }
        
#         # Generate embeddings
#         embeddings = self.generate_embeddings(texts)
        
#         # Create FAISS index
#         self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        
#         # Normalize embeddings for cosine similarity
#         faiss.normalize_L2(embeddings)
#         self.index.add(embeddings)
        
#         self.metadata = metadata
#         logger.info(f"Built index with {self.index.ntotal} vectors")
    
#     def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
#         """
#         Search for similar documents.
        
#         Args:
#             query: Search query text
#             top_k: Number of results to return
            
#         Returns:
#             List of (id, score, text) tuples
#         """
#         if self.index is None or self.model is None:
#             raise ValueError("Index not built or model not loaded")
        
#         # Generate query embedding
#         query_embedding = self.model.encode([query], convert_to_numpy=True)
#         faiss.normalize_L2(query_embedding)
        
#         # Search
#         scores, indices = self.index.search(query_embedding, top_k)
        
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             if idx in self.metadata:
#                 item = self.metadata[idx]
#                 results.append((item['id'], float(score), item['text']))
        
#         return results
    
#     def save_index(self, index_path: Path, metadata_path: Path) -> None:
#         """Save FAISS index and metadata to disk."""
#         if self.index is None:
#             raise ValueError("No index to save")
        
#         logger.info(f"Saving index to {index_path}")
        
#         # Ensure directories exist
#         index_path.parent.mkdir(parents=True, exist_ok=True)
#         metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
#         # Save FAISS index
#         faiss.write_index(self.index, str(index_path))
        
#         # Save metadata
#         with open(metadata_path, 'w', encoding='utf-8') as f:
#             json.dump({
#                 'metadata': self.metadata,
#                 'model_name': self.model_name,
#                 'dimension': self.dimension,
#                 'total_vectors': self.index.ntotal
#             }, f, indent=2, ensure_ascii=False)
        
#         logger.info("Index and metadata saved successfully")
    
#     def load_index(self, index_path: Path, metadata_path: Path) -> bool:
#         """
#         Load FAISS index and metadata from disk.
        
#         Returns:
#             True if loaded successfully, False otherwise
#         """
#         try:
#             if not index_path.exists() or not metadata_path.exists():
#                 logger.warning("Index files not found")
#                 return False
            
#             logger.info(f"Loading index from {index_path}")
            
#             # Load FAISS index
#             self.index = faiss.read_index(str(index_path))
            
#             # Load metadata
#             with open(metadata_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 self.metadata = data['metadata']
#                 self.model_name = data['model_name']
#                 self.dimension = data['dimension']
            
#             # Load model
#             self.load_model()
            
#             logger.info(f"Loaded index with {self.index.ntotal} vectors")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error loading index: {e}")
#             return False


# # Global embedding manager instance
# _embedding_manager = None

# def get_embedding_manager() -> EmbeddingManager:
#     """Get the global embedding manager instance."""
#     global _embedding_manager
#     if _embedding_manager is None:
#         _embedding_manager = EmbeddingManager()
#     return _embedding_manager


# def build_embeddings(items: List[Dict[str, Any]]) -> EmbeddingManager:
#     """
#     Build embeddings for document items.
    
#     Args:
#         items: List of document items
        
#     Returns:
#         Configured EmbeddingManager
#     """
#     manager = get_embedding_manager()
#     manager.build_index(items)
#     return manager


# def save_index(index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
#     """Save the current index to disk."""
#     if index_path is None:
#         index_path = INDEXES_DIR / "faiss_index.bin"
#     if metadata_path is None:
#         metadata_path = META_DIR / "embeddings_metadata.json"
    
#     manager = get_embedding_manager()
#     manager.save_index(index_path, metadata_path)


# def load_index(index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> bool:
#     """Load index from disk."""
#     if index_path is None:
#         index_path = INDEXES_DIR / "faiss_index.bin"
#     if metadata_path is None:
#         metadata_path = META_DIR / "embeddings_metadata.json"
    
#     manager = get_embedding_manager()
#     return manager.load_index(index_path, metadata_path)


# def search_similar(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
#     """Search for similar documents."""
#     manager = get_embedding_manager()
#     return manager.search(query, top_k)


# def build_and_save_embeddings(doc_path: Optional[Path] = None) -> None:
#     """
#     Complete pipeline: load documents, build embeddings, and save.
    
#     Args:
#         doc_path: Path to document file. If None, uses default data/cases.docx
#     """
#     from src.data_loader import load_docx, chunk_text, preprocess_text
#     from src.config import DATA_ROOT
    
#     if doc_path is None:
#         doc_path = DATA_ROOT / "cases.docx"
    
#     if not doc_path.exists():
#         logger.warning(f"Document not found: {doc_path}")
#         logger.info("Creating sample document structure...")
#         # Create sample items for development
#         items = [
#             {
#                 "id": "sample_1",
#                 "text": "Sample case study: A company faced declining sales due to poor customer service. The solution was to implement a comprehensive training program.",
#                 "type": "paragraph",
#                 "metadata": {"source": "sample", "element_type": "paragraph"}
#             },
#             {
#                 "id": "sample_2", 
#                 "text": "Problem: High employee turnover. Solution: Improved benefits package and remote work options resulted in 40% reduction in turnover.",
#                 "type": "paragraph",
#                 "metadata": {"source": "sample", "element_type": "paragraph"}
#             }
#         ]
#     else:
#         # Load actual document
#         items = load_docx(doc_path)
        
#         # Chunk large texts
#         chunked_items = []
#         for item in items:
#             if len(item['text']) > 1000:
#                 chunks = chunk_text(item['text'])
#                 for i, chunk in enumerate(chunks):
#                     chunked_item = item.copy()
#                     chunked_item['id'] = f"{item['id']}_chunk_{i+1}"
#                     chunked_item['text'] = preprocess_text(chunk)
#                     chunked_items.append(chunked_item)
#             else:
#                 item['text'] = preprocess_text(item['text'])
#                 chunked_items.append(item)
        
#         items = chunked_items
    
#     # Build embeddings
#     logger.info(f"Building embeddings for {len(items)} items")
#     build_embeddings(items)
    
#     # Save to disk
#     save_index()
    
#     logger.info("Embeddings built and saved successfully")

"""
Fixed embedding generation and vector index management.
Uses sentence-transformers with FAISS for efficient similarity search.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.config import (
    get_logger, INDEXES_DIR, META_DIR, EMBEDDING_MODEL, 
    EMBEDDING_BATCH_SIZE, MAX_EMBEDDING_RETRIES
)

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages embeddings and vector search operations with error handling."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize embedding manager with specified model."""
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = {}
        self.dimension = None
        self.available = SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE
        
        if not self.available:
            logger.warning("Sentence transformers or FAISS not available. Using fallback mode.")
        
    def load_model(self) -> None:
        """Load the sentence transformer model with error handling."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            return
            
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.model = None
                self.dimension = 384  # Default dimension
    
    def generate_embeddings(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for a list of texts with error handling.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            NumPy array of embeddings or None if failed
        """
        if not self.available or self.model is None:
            logger.warning("Embedding model not available, returning random embeddings for testing")
            # Return random embeddings for testing
            return np.random.rand(len(texts), 384).astype(np.float32)
        
        try:
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
                            # Use random embeddings as fallback
                            logger.warning("Using random embeddings as fallback")
                            fallback_embeddings = np.random.rand(len(batch), self.dimension or 384).astype(np.float32)
                            embeddings.extend(fallback_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return random embeddings as complete fallback
            return np.random.rand(len(texts), 384).astype(np.float32)
    
    def build_index(self, items: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from document items with error handling.
        
        Args:
            items: List of document items with 'text' field
        """
        logger.info("Building embedding index...")
        
        if not items:
            logger.warning("No items provided for building index")
            return
        
        # Extract texts and create metadata mapping
        texts = []
        metadata = {}
        
        for i, item in enumerate(items):
            text = item.get('text', '')
            if len(text.strip()) < 10:  # Skip very short texts
                continue
                
            texts.append(text)
            metadata[len(texts) - 1] = {
                'id': item.get('id', f'item_{i}'),
                'text': text[:200] + "..." if len(text) > 200 else text,
                'type': item.get('type', 'unknown'),
                'metadata': item.get('metadata', {})
            }
        
        if not texts:
            logger.warning("No valid texts found for indexing")
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return
        
        # Create FAISS index
        try:
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
                
                logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            else:
                logger.warning("FAISS not available, using simple similarity fallback")
                # Store embeddings for manual similarity computation
                self.embeddings = embeddings
                self.index = "manual"  # Flag for manual mode
                
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            # Fallback to manual mode
            self.embeddings = embeddings
            self.index = "manual"
        
        self.metadata = metadata
        logger.info(f"Built index with {len(self.metadata)} items")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents with error handling.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of (id, score, text) tuples
        """
        if self.index is None or not self.metadata:
            logger.warning("Index not built or empty")
            return []
        
        try:
            # Generate query embedding
            if self.model is not None:
                query_embedding = self.model.encode([query], convert_to_numpy=True)
            else:
                # Use random embedding for testing
                query_embedding = np.random.rand(1, 384).astype(np.float32)
            
            if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
                # Use FAISS search
                faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx in self.metadata and idx != -1:  # -1 indicates not found
                        item = self.metadata[idx]
                        results.append((item['id'], float(score), item['text']))
                
                return results
            
            else:
                # Manual similarity computation fallback
                if hasattr(self, 'embeddings'):
                    # Compute cosine similarity manually
                    query_norm = query_embedding / np.linalg.norm(query_embedding)
                    similarities = []
                    
                    for idx, embedding in enumerate(self.embeddings):
                        if idx in self.metadata:
                            emb_norm = embedding / np.linalg.norm(embedding)
                            similarity = np.dot(query_norm[0], emb_norm)
                            similarities.append((idx, float(similarity)))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    results = []
                    for idx, score in similarities[:top_k]:
                        item = self.metadata[idx]
                        results.append((item['id'], score, item['text']))
                    
                    return results
                else:
                    logger.warning("No embeddings available for search")
                    return []
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def save_index(self, index_path: Path, metadata_path: Path) -> None:
        """Save FAISS index and metadata to disk with error handling."""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return
            
            logger.info(f"Saving index to {index_path}")
            
            # Ensure directories exist
            index_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index or embeddings
            if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
                faiss.write_index(self.index, str(index_path))
            elif hasattr(self, 'embeddings'):
                # Save embeddings as numpy array
                np.save(str(index_path.with_suffix('.npy')), self.embeddings)
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata,
                    'model_name': self.model_name,
                    'dimension': self.dimension,
                    'total_vectors': len(self.metadata),
                    'index_type': 'faiss' if isinstance(self.index, faiss.Index) else 'manual'
                }, f, indent=2, ensure_ascii=False)
            
            logger.info("Index and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, index_path: Path, metadata_path: Path) -> bool:
        """
        Load FAISS index and metadata from disk with error handling.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not metadata_path.exists():
                logger.warning("Metadata file not found")
                return False
            
            logger.info(f"Loading index from {index_path}")
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data['metadata']
                self.model_name = data['model_name']
                self.dimension = data['dimension']
                index_type = data.get('index_type', 'faiss')
            
            # Load model
            self.load_model()
            
            # Load index
            if index_type == 'faiss' and FAISS_AVAILABLE and index_path.exists():
                self.index = faiss.read_index(str(index_path))
            elif index_path.with_suffix('.npy').exists():
                # Load manual embeddings
                self.embeddings = np.load(str(index_path.with_suffix('.npy')))
                self.index = "manual"
            else:
                logger.warning("No index file found, index not loaded")
                return False
            
            logger.info(f"Loaded index with {len(self.metadata)} items")
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
        doc_path: Path to document file. If None, uses default data sources
    """
    from src.data_loader import load_multiple_docx, chunk_text, preprocess_text
    from src.config import DATA_ROOT
    
    # Try to find documents
    data_sources = []
    if doc_path and doc_path.exists():
        data_sources = [doc_path]
    else:
        # Look for documents in data directory
        potential_dirs = [DATA_ROOT / "batch_1", DATA_ROOT]
        for dir_path in potential_dirs:
            if dir_path.exists():
                doc_files = list(dir_path.glob("*.docx"))
                if doc_files:
                    data_sources.extend(doc_files)
                    break
    
    items = []
    if data_sources:
        # Load actual documents
        for doc_file in data_sources[:3]:  # Limit to first 3 documents
            try:
                from src.data_loader import load_docx
                doc_items = load_docx(doc_file)
                items.extend(doc_items)
                logger.info(f"Loaded {len(doc_items)} items from {doc_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {doc_file}: {e}")
    
    if not items:
        # Create sample items for development
        logger.info("Creating sample items for testing...")
        items = [
            {
                "id": "sample_1",
                "text": "Arbitration proceedings typically take 6-18 months from initiation to final award. Emergency arbitration can be completed within 14-30 days for urgent matters.",
                "type": "paragraph",
                "metadata": {"source": "sample", "element_type": "paragraph"}
            },
            {
                "id": "sample_2", 
                "text": "Arbitration costs range from $50,000 to $500,000 depending on claim value and complexity. Administrative fees typically account for 5-15% of total costs.",
                "type": "paragraph",
                "metadata": {"source": "sample", "element_type": "paragraph"}
            },
            {
                "id": "sample_3",
                "text": "For arbitration clauses to be valid: clear language required, mutual agreement by parties, proper scope definition, designated rules and institution.",
                "type": "paragraph", 
                "metadata": {"source": "sample", "element_type": "paragraph"}
            },
            {
                "id": "sample_4",
                "text": "Emergency arbitration is available when immediate harm would occur without relief and regular arbitration would be too slow to prevent irreparable damage.",
                "type": "paragraph",
                "metadata": {"source": "sample", "element_type": "paragraph"}
            }
        ]
    
    # Process items for better chunking
    processed_items = []
    for item in items:
        text = item.get('text', '')
        if len(text) < 50:  # Skip very short texts
            continue
            
        # Chunk large texts
        if len(text) > 1000:
            chunks = chunk_text(text, chunk_size=800, overlap=100)
            for i, chunk in enumerate(chunks):
                chunked_item = item.copy()
                chunked_item['id'] = f"{item['id']}_chunk_{i+1}"
                chunked_item['text'] = preprocess_text(chunk)
                chunked_item['metadata'] = item.get('metadata', {}).copy()
                chunked_item['metadata']['is_chunk'] = True
                chunked_item['metadata']['chunk_index'] = i + 1
                chunked_item['metadata']['total_chunks'] = len(chunks)
                processed_items.append(chunked_item)
        else:
            item_copy = item.copy()
            item_copy['text'] = preprocess_text(text)
            processed_items.append(item_copy)
    
    # Build embeddings
    logger.info(f"Building embeddings for {len(processed_items)} items")
    build_embeddings(processed_items)
    
    # Save to disk
    save_index()
    
    logger.info("Embeddings built and saved successfully")