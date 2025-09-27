"""
Tests for embedding functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.embeddings import EmbeddingManager, build_embeddings, save_index, load_index


class TestEmbeddingManager:
    """Test cases for EmbeddingManager."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = EmbeddingManager()
        self.test_items = [
            {
                'id': 'test_1',
                'text': 'This is a test document about customer service problems.',
                'type': 'paragraph',
                'metadata': {'source': 'test'}
            },
            {
                'id': 'test_2', 
                'text': 'Another test document discussing solutions and approaches.',
                'type': 'paragraph',
                'metadata': {'source': 'test'}
            }
        ]
    
    @patch('src.embeddings.SentenceTransformer')
    def test_load_model(self, mock_transformer):
        """Test model loading."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        self.manager.load_model()
        
        assert self.manager.model is not None
        assert self.manager.dimension == 384
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
    
    @patch('src.embeddings.SentenceTransformer')
    def test_generate_embeddings(self, mock_transformer):
        """Test embedding generation."""
        mock_model = Mock()
        mock_embeddings = np.random.rand(2, 384)  # 2 texts, 384 dimensions
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        texts = ['Text 1', 'Text 2']
        embeddings = self.manager.generate_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)
        mock_model.encode.assert_called_once()
    
    @patch('src.embeddings.faiss')
    @patch('src.embeddings.SentenceTransformer')
    def test_build_index(self, mock_transformer, mock_faiss):
        """Test index building."""
        # Mock transformer
        mock_model = Mock()
        mock_embeddings = np.random.rand(2, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        # Mock FAISS
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        self.manager.build_index(self.test_items)
        
        assert self.manager.index is not None
        assert len(self.manager.metadata) == 2
        mock_faiss.IndexFlatIP.assert_called_once_with(384)
    
    @patch('src.embeddings.faiss')
    @patch('src.embeddings.SentenceTransformer')
    def test_search(self, mock_transformer, mock_faiss):
        """Test similarity search."""
        # Setup mocks
        mock_model = Mock()
        mock_query_embedding = np.random.rand(1, 384)
        mock_model.encode.return_value = mock_query_embedding
        mock_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_scores = np.array([[0.8, 0.6]])
        mock_indices = np.array([[0, 1]])
        mock_index.search.return_value = (mock_scores, mock_indices)
        
        # Setup manager state
        self.manager.model = mock_model
        self.manager.index = mock_index
        self.manager.metadata = {
            0: {'id': 'test_1', 'text': 'First document'},
            1: {'id': 'test_2', 'text': 'Second document'}
        }
        
        results = self.manager.search("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == 'test_1'  # First result ID
        assert results[0][1] == 0.8  # First result score
    
    def test_save_and_load_index(self):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            index_path = tmp_path / "test_index.bin"
            metadata_path = tmp_path / "test_metadata.json"
            
            # Mock index for saving
            with patch('src.embeddings.faiss') as mock_faiss:
                mock_index = Mock()
                mock_index.ntotal = 2
                self.manager.index = mock_index
                self.manager.metadata = {'0': {'id': 'test_1', 'text': 'Test'}}
                self.manager.model_name = 'test-model'
                self.manager.dimension = 384
                
                # Test saving
                self.manager.save_index(index_path, metadata_path)
                
                mock_faiss.write_index.assert_called_once()
                assert metadata_path.exists()
            
            # Test loading
            with patch('src.embeddings.faiss') as mock_faiss, \
                 patch('src.embeddings.SentenceTransformer') as mock_transformer:
                
                mock_index = Mock()
                mock_faiss.read_index.return_value = mock_index
                
                mock_model = Mock()
                mock_transformer.return_value = mock_model
                
                new_manager = EmbeddingManager()
                success = new_manager.load_index(index_path, metadata_path)
                
                assert success
                assert new_manager.index is not None
                assert new_manager.metadata is not None


class TestEmbeddingFunctions:
    """Test module-level embedding functions."""
    
    @patch('src.embeddings.get_embedding_manager')
    def test_build_embeddings(self, mock_get_manager):
        """Test build_embeddings function."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        test_items = [{'id': 'test', 'text': 'test text'}]
        result = build_embeddings(test_items)
        
        assert result == mock_manager
        mock_manager.build_index.assert_called_once_with(test_items)
    
    @patch('src.embeddings.get_embedding_manager')
    def test_save_index(self, mock_get_manager):
        """Test save_index function."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        save_index()
        
        mock_manager.save_index.assert_called_once()
    
    @patch('src.embeddings.get_embedding_manager')
    def test_load_index(self, mock_get_manager):
        """Test load_index function."""
        mock_manager = Mock()
        mock_manager.load_index.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = load_index()
        
        assert result is True
        mock_manager.load_index.assert_called_once()
    
    @patch('src.embeddings.get_embedding_manager')
    def test_search_similar(self, mock_get_manager):
        """Test search_similar function."""
        mock_manager = Mock()
        mock_results = [('doc_1', 0.9, 'text 1'), ('doc_2', 0.7, 'text 2')]
        mock_manager.search.return_value = mock_results
        mock_get_manager.return_value = mock_manager
        
        from src.embeddings import search_similar
        results = search_similar("test query", top_k=2)
        
        assert results == mock_results
        mock_manager.search.assert_called_once_with("test query", 2)


class TestEmbeddingIntegration:
    """Integration tests for embedding functionality."""
    
    @patch('src.embeddings.SentenceTransformer')
    @patch('src.embeddings.faiss')
    def test_full_pipeline(self, mock_faiss, mock_transformer):
        """Test complete embedding pipeline."""
        # Setup mocks
        mock_model = Mock()
        mock_embeddings = np.random.rand(2, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Test data
        items = [
            {'id': 'doc1', 'text': 'First test document', 'type': 'test'},
            {'id': 'doc2', 'text': 'Second test document', 'type': 'test'}
        ]
        
        # Build embeddings
        manager = build_embeddings(items)
        
        # Verify the process
        assert manager.index is not None
        assert len(manager.metadata) == 2
        mock_transformer.assert_called()
        mock_faiss.IndexFlatIP.assert_called_with(384)


if __name__ == "__main__":
    pytest.main([__file__])