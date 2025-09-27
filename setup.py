#!/usr/bin/env python3
"""
Setup script for Case Study Chatbot PoC.
Processes documents, builds indexes, and validates the system.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from src.config import get_logger, DATA_ROOT, STORAGE_ROOT
from src.data_loader import load_multiple_docx, get_document_summary, chunk_text, preprocess_text
from src.embeddings import build_and_save_embeddings, load_index, EmbeddingManager
from src.knowledge_graph import build_enhanced_kg, save_enhanced_kg, load_enhanced_kg
from src.agent import get_agent
from src.retriever import get_retriever

logger = get_logger(__name__)

def validate_environment():
    """Validate the environment and configuration."""
    logger.info("Validating environment...")
    
    # Check API keys
    from src.config import GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER
    
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")
    elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env file.")
    
    # Check data directory
    data_dirs = [DATA_ROOT / "batch_1", DATA_ROOT]
    valid_dir = None
    
    for data_dir in data_dirs:
        if data_dir.exists() and list(data_dir.glob("*.docx")):
            valid_dir = data_dir
            break
    
    if not valid_dir:
        raise ValueError(f"No .docx files found in {DATA_ROOT} or {DATA_ROOT}/batch_1")
    
    logger.info(f"Using data directory: {valid_dir}")
    return valid_dir

def process_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """Process all documents in the data directory."""
    logger.info("Processing documents...")
    
    # Get document summary
    summary = get_document_summary(data_dir)
    logger.info(f"Found {summary['total_documents']} documents ({summary['total_size_mb']:.2f} MB)")
    
    for doc_info in summary['documents']:
        logger.info(f"  - {doc_info['name']}: {doc_info['paragraphs']} paragraphs, {doc_info['tables']} tables")
    
    # Load all documents
    items = load_multiple_docx(data_dir)
    logger.info(f"Extracted {len(items)} raw items")
    
    # Enhanced preprocessing
    processed_items = []
    for item in items:
        # Clean text
        clean_text = preprocess_text(item['text'])
        
        # Skip very short items
        if len(clean_text) < 50:
            continue
        
        # Chunk if necessary
        if len(clean_text) > 1000:
            chunks = chunk_text(clean_text, chunk_size=800, overlap=100)
            for i, chunk in enumerate(chunks):
                chunk_item = item.copy()
                chunk_item['id'] = f"{item['id']}_chunk_{i+1}"
                chunk_item['text'] = chunk
                chunk_item['metadata']['is_chunk'] = True
                chunk_item['metadata']['chunk_index'] = i + 1
                chunk_item['metadata']['total_chunks'] = len(chunks)
                processed_items.append(chunk_item)
        else:
            item['text'] = clean_text
            processed_items.append(item)
    
    logger.info(f"After processing: {len(processed_items)} items")
    return processed_items

def build_embeddings_index(items: List[Dict[str, Any]]) -> bool:
    """Build and save embeddings index."""
    logger.info("Building embeddings index...")
    
    try:
        # Check if index already exists and is recent
        from src.config import INDEXES_DIR, META_DIR
        index_path = INDEXES_DIR / "faiss_index.bin"
        metadata_path = META_DIR / "embeddings_metadata.json"
        
        # Try to load existing index
        manager = EmbeddingManager()
        if manager.load_index(index_path, metadata_path):
            logger.info("Loaded existing embeddings index")
            
            # Verify index quality
            test_query = "customer service problem"
            results = manager.search(test_query, top_k=3)
            if len(results) >= 2:
                logger.info("Existing index appears to be working correctly")
                return True
            else:
                logger.warning("Existing index seems incomplete, rebuilding...")
        
        # Build new index
        logger.info(f"Building new embeddings index for {len(items)} items...")
        manager.build_index(items)
        manager.save_index(index_path, metadata_path)
        
        # Test the new index
        test_results = manager.search("customer service", top_k=3)
        logger.info(f"Index test: found {len(test_results)} results for 'customer service'")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build embeddings index: {e}")
        return False

def build_knowledge_graph_index(items: List[Dict[str, Any]]) -> bool:
    """Build and save knowledge graph."""
    logger.info("Building knowledge graph...")
    
    try:
        # Try enhanced KG first
        try:
            kg = build_enhanced_kg()
            save_enhanced_kg(kg)
            
            # Test the KG
            from src.knowledge_graph import query_enhanced_kg
            test_results = query_enhanced_kg("customer service problem", kg, max_results=3)
            logger.info(f"Enhanced KG test: found {len(test_results)} results")
            
            if test_results:
                logger.info("Enhanced knowledge graph built successfully")
                return True
                
        except Exception as e:
            logger.warning(f"Enhanced KG failed: {e}, falling back to basic KG")
        
        # Fallback to basic KG
        from src.knowledge_graph import build_kg, save_kg
        kg = build_kg()
        save_kg(kg)
        
        logger.info(f"Basic knowledge graph built with {kg.number_of_nodes()} nodes")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        return False

def test_system_integration() -> bool:
    """Test the complete system integration."""
    logger.info("Testing system integration...")
    
    test_queries = [
        "What are the main customer service problems?",
        "How can companies improve employee satisfaction?",
        "What solutions work best for reducing costs?",
        "Analyze the effectiveness of training programs"
    ]
    
    try:
        agent = get_agent()
        retriever = get_retriever()
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Test {i}: {query}")
            
            # Test retrieval
            context = retriever.retrieve_context(query)
            logger.info(f"  Retrieved {context['total_sources']} sources")
            
            # Test agent response
            response = agent.answer_query(query)
            logger.info(f"  Response length: {len(response.answer)} chars")
            logger.info(f"  Confidence: {response.confidence:.2f}")
            
            if len(response.answer) < 50:
                logger.warning(f"  Response seems too short: {response.answer}")
            
            time.sleep(1)  # Brief pause between tests
        
        logger.info("System integration tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System integration test failed: {e}")
        return False

def create_sample_data():
    """Create sample data if no documents are found."""
    logger.info("Creating sample data...")
    
    sample_dir = DATA_ROOT / "batch_1"
    sample_dir.mkdir(exist_ok=True)
    
    # This would require python-docx to create actual .docx files
    # For now, we'll create the directory structure
    logger.info(f"Sample data directory created at {sample_dir}")
    logger.info("Please add your .docx files to this directory")

def main():
    """Main setup function."""
    print("🚀 Case Study Chatbot PoC Setup")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Step 1: Validate environment
        data_dir = validate_environment()
        print("✅ Environment validation passed")
        
        # Step 2: Process documents
        items = process_documents(data_dir)
        print(f"✅ Processed {len(items)} document items")
        
        # Step 3: Build embeddings
        if build_embeddings_index(items):
            print("✅ Embeddings index built successfully")
        else:
            print("❌ Failed to build embeddings index")
            return False
        
        # Step 4: Build knowledge graph
        if build_knowledge_graph_index(items):
            print("✅ Knowledge graph built successfully")
        else:
            print("❌ Failed to build knowledge graph")
            return False
        
        # Step 5: Test system
        if test_system_integration():
            print("✅ System integration tests passed")
        else:
            print("⚠️  System integration tests had issues")
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Setup completed in {elapsed_time:.2f} seconds")
        print("\nYou can now run the chatbot with:")
        print("  python src/ui.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()