import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os

from src.data_loader import load_docx, chunk_text, extract_table_content, preprocess_text, extract_qa_pairs, load_multiple_docx, get_document_summary


class TestDocumentLoader:
    """Test cases for document loading functionality."""
    
    def test_chunk_text_small_input(self):
        """Test chunking with input smaller than chunk size."""
        text = "This is a small text."
        chunks = chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_large_input(self):
        """Test chunking with input larger than chunk size."""
        text = "A " * 600  # 1200 characters
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 500 for chunk in chunks)
    
    def test_chunk_text_sentence_boundary(self):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        
        assert len(chunks) >= 2
        assert all(chunk.endswith(".") for chunk in chunks[:-1])  # Ensure chunks (except last) end with sentence
        assert "First sentence." in chunks[0]  # Verify first chunk contains first sentence
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        dirty_text = "  This   has    extra   spaces!  \n\t  "
        clean_text = preprocess_text(dirty_text)
        
        assert clean_text == "This has extra spaces!"
        assert not clean_text.startswith(" ")
        assert not clean_text.endswith(" ")
    
    @patch('src.data_loader.Document')
    def test_load_docx_file_not_found(self, mock_document):
        """Test behavior when document file doesn't exist."""
        non_existent_path = Path("non_existent.docx")
        
        with pytest.raises(FileNotFoundError):
            load_docx(non_existent_path)
    
    @patch('src.data_loader.Document')
    def test_load_docx_with_paragraphs(self, mock_document):
        """Test loading document with paragraphs."""
        # Mock document structure
        mock_para = Mock()
        mock_para.text = "This is a test paragraph with sufficient length for processing."
        
        mock_element = Mock()
        mock_element.tag = "paragraph"
        
        mock_doc_instance = Mock()
        mock_doc_instance.element.body = [mock_element]
        mock_doc_instance.paragraphs = [mock_para]
        mock_doc_instance.tables = []
        
        mock_document.return_value = mock_doc_instance
        
        # Create a temporary file to satisfy path.exists()
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            items = load_docx(tmp_path)
            
            assert len(items) >= 1
            assert items[0]['type'] == 'paragraph'
            assert items[0]['text'] == preprocess_text(mock_para.text)
            mock_document.assert_called_once_with(str(tmp_path))
            
        finally:
            os.unlink(tmp_path)
    
    def test_extract_table_content_empty_table(self):
        """Test extracting content from an empty table."""
        mock_table = Mock()
        mock_table.rows = []
        
        items = extract_table_content(mock_table, "test.docx", 1)
        assert items == []
    
    def test_extract_table_content_with_data(self):
        """Test extracting content from a table with data."""
        # Mock table structure
        mock_cell1 = Mock()
        mock_cell1.text = "Header 1"
        mock_cell2 = Mock()
        mock_cell2.text = "Header 2"
        
        mock_cell3 = Mock()
        mock_cell3.text = "Data 1 with sufficient length for processing and testing"
        mock_cell4 = Mock()
        mock_cell4.text = "Data 2 with sufficient length for processing and testing"
        
        mock_header_row = Mock()
        mock_header_row.cells = [mock_cell1, mock_cell2]
        
        mock_data_row = Mock()
        mock_data_row.cells = [mock_cell3, mock_cell4]
        
        mock_table = Mock()
        mock_table.rows = [mock_header_row, mock_data_row]
        
        items = extract_table_content(mock_table, "test.docx", 1)
        
        assert len(items) == 1
        assert items[0]['type'] == 'table_row'
        assert 'Header 1: Data 1' in items[0]['text']
        assert 'Header 2: Data 2' in items[0]['text']

    @patch('src.data_loader.load_docx')
    def test_load_multiple_docx(self, mock_load_docx):
        """Test loading multiple Word documents from a directory."""
        # Mock directory and files
        mock_dir = Path(tempfile.mkdtemp())
        mock_files = [mock_dir / "doc1.docx", mock_dir / "doc2.docx"]
        
        with patch.object(Path, 'glob', return_value=mock_files):
            mock_load_docx.side_effect = [
                [{'id': 'para_1', 'type': 'paragraph', 'text': 'Text 1', 'metadata': {'source': 'doc1.docx'}}],
                [{'id': 'para_1', 'type': 'paragraph', 'text': 'Text 2', 'metadata': {'source': 'doc2.docx'}}]
            ]
            
            items = load_multiple_docx(mock_dir)
            
            assert len(items) == 2
            assert items[0]['id'] == 'doc1_para_1'
            assert items[0]['metadata']['document_name'] == 'doc1.docx'
            assert items[0]['metadata']['document_index'] == 1
            assert items[1]['id'] == 'doc2_para_1'
            assert items[1]['metadata']['document_name'] == 'doc2.docx'
            assert items[1]['metadata']['document_index'] == 2
    
    @patch('src.data_loader.Document')
    def test_get_document_summary(self, mock_document):
        """Test getting summary of documents in a directory."""
        # Mock directory and files
        mock_dir = Path(tempfile.mkdtemp())
        mock_files = [mock_dir / "doc1.docx", mock_dir / "doc2.docx"]
        
        # Mock document structure
        mock_doc1 = Mock()
        mock_doc1.paragraphs = [Mock(text="Text 1"), Mock(text="")]
        mock_doc1.tables = [Mock(rows=[Mock(), Mock()])]
        
        mock_doc2 = Mock()
        mock_doc2.paragraphs = [Mock(text="Text 2"), Mock(text="Text 3")]
        mock_doc2.tables = []
        
        mock_document.side_effect = [mock_doc1, mock_doc2]
        
        with patch.object(Path, 'glob', return_value=mock_files):
            with patch.object(Path, 'stat', return_value=Mock(st_size=1024*1024)):  # 1MB per file
                summary = get_document_summary(mock_dir)
                
                assert summary['total_documents'] == 2
                assert summary['document_names'] == ['doc1.docx', 'doc2.docx']
                assert summary['total_size_mb'] == 2.0
                assert len(summary['documents']) == 2
                assert summary['documents'][0]['name'] == 'doc1.docx'
                assert summary['documents'][0]['paragraphs'] == 1
                assert summary['documents'][0]['tables'] == 1
                assert summary['documents'][0]['estimated_items'] == 3
                assert summary['documents'][1]['name'] == 'doc2.docx'
                assert summary['documents'][1]['paragraphs'] == 2
                assert summary['documents'][1]['tables'] == 0
                assert summary['documents'][1]['estimated_items'] == 2


class TestQAExtraction:
    """Test Q&A pair extraction functionality."""
    
    def test_extract_qa_pairs_simple(self):
        """Test extraction of simple Q&A pairs."""
        items = [
            {
                'id': 'q1',
                'text': 'Q: What is the main problem?',
                'metadata': {'source': 'test'}
            },
            {
                'id': 'a1', 
                'text': 'A: The main problem is customer dissatisfaction.',
                'metadata': {'source': 'test'}
            }
        ]
        
        qa_pairs = extract_qa_pairs(items)
        
        assert len(qa_pairs) == 1
        assert qa_pairs[0]['question'] == 'What is the main problem?'
        assert qa_pairs[0]['answer'] == 'The main problem is customer dissatisfaction.'
    
    def test_extract_qa_pairs_no_matches(self):
        """Test extraction when no Q&A patterns are found."""
        items = [
            {
                'id': 'p1',
                'text': 'This is just a regular paragraph.',
                'metadata': {'source': 'test'}
            }
        ]
        
        qa_pairs = extract_qa_pairs(items)
        assert len(qa_pairs) == 0


if __name__ == "__main__":
    pytest.main([__file__])