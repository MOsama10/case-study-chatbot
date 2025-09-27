#!/usr/bin/env python3
"""
Diagnostic script to find import issues in the Case Study Chatbot.
"""

import sys
import traceback
from pathlib import Path

def test_individual_imports():
    """Test each module individually to find the problematic import."""
    
    modules_to_test = [
        'src.config',
        'src.data_loader', 
        'src.embeddings',
        'src.knowledge_graph',
        'src.retriever',
        'src.agent',
        'src.chatbot',
        'src.ui'
    ]
    
    print("🔍 Testing individual module imports...")
    print("=" * 50)
    
    for module_name in modules_to_test:
        try:
            print(f"Testing {module_name}...", end=" ")
            __import__(module_name)
            print("✅ OK")
        except Exception as e:
            print("❌ FAILED")
            print(f"  Error: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            print()

def test_knowledge_graph_functions():
    """Test knowledge graph functions specifically."""
    print("\n🕸️ Testing Knowledge Graph functions...")
    print("=" * 50)
    
    try:
        from src.knowledge_graph import build_kg
        print("✅ build_kg imported")
    except Exception as e:
        print(f"❌ build_kg failed: {e}")
    
    try:
        from src.knowledge_graph import load_kg
        print("✅ load_kg imported")
    except Exception as e:
        print(f"❌ load_kg failed: {e}")
    
    try:
        from src.knowledge_graph import save_kg
        print("✅ save_kg imported")
    except Exception as e:
        print(f"❌ save_kg failed: {e}")
    
    try:
        from src.knowledge_graph import get_knowledge_graph
        print("✅ get_knowledge_graph imported")
    except Exception as e:
        print(f"❌ get_knowledge_graph failed: {e}")
    
    try:
        from src.knowledge_graph import query_kg
        print("✅ query_kg imported")
    except Exception as e:
        print(f"❌ query_kg failed: {e}")

def test_retriever_imports():
    """Test retriever imports specifically."""
    print("\n🔄 Testing Retriever imports...")
    print("=" * 50)
    
    try:
        print("Importing retriever module...", end=" ")
        import src.retriever
        print("✅ OK")
        
        print("Testing get_knowledge_graph in retriever...", end=" ")
        # Check if retriever is trying to import get_knowledge_graph
        retriever_code = Path("src/retriever.py").read_text()
        if "get_knowledge_graph" in retriever_code:
            print("⚠️ Found reference")
            print("  Retriever is trying to import get_knowledge_graph")
        else:
            print("✅ No reference found")
            
    except Exception as e:
        print("❌ FAILED")
        print(f"  Error: {e}")

def test_agent_imports():
    """Test agent imports specifically."""
    print("\n🤖 Testing Agent imports...")
    print("=" * 50)
    
    try:
        print("Importing agent module...", end=" ")
        import src.agent
        print("✅ OK")
        
        print("Testing get_retriever in agent...", end=" ")
        from src.agent import get_agent
        print("✅ OK")
        
    except Exception as e:
        print("❌ FAILED")
        print(f"  Error: {e}")

def check_circular_imports():
    """Check for potential circular import issues."""
    print("\n🔄 Checking for circular imports...")
    print("=" * 50)
    
    files_to_check = [
        "src/knowledge_graph.py",
        "src/retriever.py", 
        "src/agent.py",
        "src/embeddings.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            content = Path(file_path).read_text()
            imports = []
            for line in content.split('\n'):
                if line.strip().startswith('from .') or line.strip().startswith('from src.'):
                    imports.append(line.strip())
            
            print(f"\n{file_path}:")
            for imp in imports[:5]:  # Show first 5 imports
                print(f"  {imp}")
            if len(imports) > 5:
                print(f"  ... and {len(imports) - 5} more")

def create_minimal_test():
    """Create a minimal test to isolate the issue."""
    print("\n🧪 Running minimal test...")
    print("=" * 50)
    
    try:
        print("Step 1: Import knowledge_graph...", end=" ")
        from src import knowledge_graph
        print("✅ OK")
        
        print("Step 2: Get function list...", end=" ")
        functions = [f for f in dir(knowledge_graph) if not f.startswith('_')]
        print("✅ OK")
        print(f"  Available functions: {functions}")
        
        print("Step 3: Test get_knowledge_graph...", end=" ")
        if hasattr(knowledge_graph, 'get_knowledge_graph'):
            kg = knowledge_graph.get_knowledge_graph()
            print(f"✅ OK (nodes: {kg.number_of_nodes()})")
        else:
            print("❌ Function not found")
            
    except Exception as e:
        print("❌ FAILED")
        print(f"  Error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

def main():
    """Run all diagnostic tests."""
    print("🚀 Case Study Chatbot - Import Diagnostics")
    print("=" * 60)
    
    # Change to project directory
    if not Path("src").exists():
        print("❌ Not in project directory. Please run from case-study-chatbot/")
        return
    
    test_individual_imports()
    test_knowledge_graph_functions()
    test_retriever_imports()
    test_agent_imports()
    check_circular_imports()
    create_minimal_test()
    
    print("\n" + "=" * 60)
    print("🔧 Diagnostic complete. Check the ❌ items above for issues to fix.")

if __name__ == "__main__":
    main()