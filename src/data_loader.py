import logging
from pathlib import Path
from typing import List, Dict, Any
from docx import Document
import re

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text by removing extra spaces and newlines.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Cleaned text string
    """
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into chunks while respecting sentence boundaries when possible.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length <= chunk_size:
            current_chunk += sentence + " "
            current_length += sentence_length + 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + sentence + " "
            current_length = len(overlap_text) + sentence_length + 1
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_table_content(table: Any, doc_name: str, table_index: int) -> List[Dict[str, Any]]:
    """
    Extract content from a Word document table.
    
    Args:
        table: Document table object
        doc_name: Name of the document
        table_index: Index of the table in the document
        
    Returns:
        List of table row items with metadata
    """
    items = []
    if not table.rows:
        return items
    
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    
    for row_idx, row in enumerate(table.rows[1:], 1):
        row_text = ""
        for idx, cell in enumerate(row.cells):
            if idx < len(headers):
                row_text += f"{headers[idx]}: {cell.text.strip()} "
        
        items.append({
            'id': f"table_{table_index}_row_{row_idx}",
            'type': 'table_row',
            'text': preprocess_text(row_text),
            'metadata': {
                'source': doc_name,
                'table_index': table_index,
                'row_index': row_idx
            }
        })
    
    return items

def load_docx(doc_path: Path) -> List[Dict[str, Any]]:
    """
    Load content from a single Word document.
    
    Args:
        doc_path: Path to the .docx file
        
    Returns:
        List of items (paragraphs and table rows) with metadata
    """
    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        raise FileNotFoundError(f"Document not found: {doc_path}")
    
    items = []
    try:
        doc = Document(str(doc_path))
        
        # Process paragraphs
        for para_idx, paragraph in enumerate(doc.paragraphs, 1):
            text = paragraph.text.strip()
            if text:
                items.append({
                    'id': f"para_{para_idx}",
                    'type': 'paragraph',
                    'text': preprocess_text(text),
                    'metadata': {
                        'source': doc_path.name,
                        'paragraph_index': para_idx
                    }
                })
        
        # Process tables
        for table_idx, table in enumerate(doc.tables, 1):
            table_items = extract_table_content(table, doc_path.name, table_idx)
            items.extend(table_items)
        
        logger.info(f"Loaded {len(items)} items from {doc_path.name}")
        return items
    
    except Exception as e:
        logger.error(f"Error loading document {doc_path.name}: {e}")
        raise

def load_multiple_docx(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load multiple Word documents from a directory.
    
    Args:
        data_dir: Directory containing .docx files
        
    Returns:
        Combined list of items from all documents
    """
    all_items = []
    docx_files = list(data_dir.glob("*.docx"))
    
    if not docx_files:
        logger.warning(f"No .docx files found in {data_dir}")
        return all_items
    
    logger.info(f"Found {len(docx_files)} Word documents to process")
    
    for i, docx_file in enumerate(docx_files, 1):
        try:
            logger.info(f"Processing document {i}/{len(docx_files)}: {docx_file.name}")
            items = load_docx(docx_file)
            
            # Add document source info to metadata
            for item in items:
                item['metadata']['document_name'] = docx_file.name
                item['metadata']['document_index'] = i
                item['id'] = f"{docx_file.stem}_{item['id']}"  # Prefix with filename
            
            all_items.extend(items)
            logger.info(f"Extracted {len(items)} items from {docx_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {docx_file.name}: {e}")
            continue
    
    logger.info(f"Total items extracted from all documents: {len(all_items)}")
    return all_items

def get_document_summary(data_dir: Path) -> Dict[str, Any]:
    """
    Get summary of all documents in the directory.
    
    Args:
        data_dir: Directory containing documents
        
    Returns:
        Summary statistics
    """
    docx_files = list(data_dir.glob("*.docx"))
    
    summary = {
        'total_documents': len(docx_files),
        'document_names': [f.name for f in docx_files],
        'total_size_mb': sum(f.stat().st_size for f in docx_files) / (1024 * 1024),
        'documents': []
    }
    
    for docx_file in docx_files:
        try:
            # Quick analysis without full processing
            doc = Document(str(docx_file))
            para_count = len([p for p in doc.paragraphs if p.text.strip()])
            table_count = len(doc.tables)
            
            doc_info = {
                'name': docx_file.name,
                'size_mb': docx_file.stat().st_size / (1024 * 1024),
                'paragraphs': para_count,
                'tables': table_count,
                'estimated_items': para_count + sum(len(table.rows) for table in doc.tables)
            }
            summary['documents'].append(doc_info)
            
        except Exception as e:
            logger.warning(f"Could not analyze {docx_file.name}: {e}")
    
    return summary

def extract_qa_pairs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract question-answer pairs from a list of items.
    
    Args:
        items: List of items with text and metadata
        
    Returns:
        List of Q&A pairs
    """
    qa_pairs = []
    i = 0
    while i < len(items) - 1:
        current_text = items[i]['text']
        next_text = items[i + 1]['text']
        
        if current_text.startswith('Q:') and next_text.startswith('A:'):
            qa_pairs.append({
                'question': preprocess_text(current_text[2:].strip()),
                'answer': preprocess_text(next_text[2:].strip()),
                'metadata': {
                    'source': items[i]['metadata']['source'],
                    'question_id': items[i]['id'],
                    'answer_id': items[i + 1]['id']
                }
            })
            i += 2
        else:
            i += 1
    
    return qa_pairs